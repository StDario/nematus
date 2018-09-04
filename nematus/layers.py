'''
Layer definitions
'''

import json
import cPickle as pkl
import numpy
from collections import OrderedDict

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from initializers import *
from util import *
from theano_util import *
from alignment_util import *

#from theano import printing

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'gru_cond_ctx': ('param_init_gru_cond_ctx', 'gru_cond_layer_ctx'),
          'gru_cond_mul_ctx': ('param_init_gru_cond_ctx', 'gru_cond_layer_mul_ctx'),
          'gru_cond_ctx_nogate': ('param_init_gru_cond_ctx_nogate', 'gru_cond_layer_ctx_nogate'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          'embedding': ('param_init_embedding_layer', 'embedding_layer')
          }


def dropout_constr(options, use_noise, trng, sampling):
    """This constructor takes care of the fact that we want different
    behaviour in training and sampling, and keeps backward compatibility:
    on older versions, activations need to be rescaled at test time;
    on newer veresions, they are rescaled at training time.
    """

    # if dropout is off, or we don't need it because we're sampling, multiply by 1
    # this is also why we make all arguments optional
    def get_layer(shape=None, dropout_probability=0, num=1):
        if num > 1:
            return theano.shared(numpy.array([1.]*num, dtype=floatX))
        else:
            return theano.shared(numpy_floatX(1.))

    if options['use_dropout']:
        # models trained with old dropout need to be rescaled at test time
        if sampling and options['model_version'] < 0.1:
            def get_layer(shape=None, dropout_probability=0, num=1):
                if num > 1:
                    return theano.shared(numpy.array([1-dropout_probability]*num, dtype=floatX))
                else:
                    return theano.shared(numpy_floatX(1-dropout_probability))
        elif not sampling:
            if options['model_version'] < 0.1:
                scaled = False
            else:
                scaled = True
            def get_layer(shape, dropout_probability=0, num=1):
                if num > 1:
                    return shared_dropout_layer((num,) + shape, use_noise, trng, 1-dropout_probability, scaled)
                else:
                    return shared_dropout_layer(shape, use_noise, trng, 1-dropout_probability, scaled)

    return get_layer


def get_layer_param(name):
    param_fn, constr_fn = layers[name]
    return eval(param_fn)

def get_layer_constr(name):
    param_fn, constr_fn = layers[name]
    return eval(constr_fn)

# dropout that will be re-used at different time steps
def shared_dropout_layer(shape, use_noise, trng, value, scaled=True):
    #re-scale dropout at training time, so we don't need to at test time
    if scaled:
        proj = tensor.switch(
            use_noise,
            trng.binomial(shape, p=value, n=1,
                                        dtype=floatX)/value,
            theano.shared(numpy_floatX(1.)))
    else:
        proj = tensor.switch(
            use_noise,
            trng.binomial(shape, p=value, n=1,
                                        dtype=floatX),
            theano.shared(numpy_floatX(value)))
    return proj

# layer normalization
# code from https://github.com/ryankiros/layer-norm
def layer_norm(x, b, s):
    _eps = numpy_floatX(1e-5)
    if x.ndim == 3:
        output = (x - x.mean(2)[:,:,None]) / tensor.sqrt((x.var(2)[:,:,None] + _eps))
        output = s[None, None, :] * output + b[None, None,:]
    else:
        output = (x - x.mean(1)[:,None]) / tensor.sqrt((x.var(1)[:,None] + _eps))
        output = s[None, :] * output + b[None,:]
    return output

def weight_norm(W, s):
    """
    Normalize the columns of a matrix
    """
    _eps = numpy_floatX(1e-5)
    W_norms = tensor.sqrt((W * W).sum(axis=0, keepdims=True) + _eps)
    W_norms_s = W_norms * s # do this first to ensure proper broadcasting
    return W / W_norms_s

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True, weight_matrix=True, bias=True, followed_by_softmax=False):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    if weight_matrix:
        params[pp(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    if bias:
       params[pp(prefix, 'b')] = numpy.zeros((nout,)).astype(floatX)

    if options['layer_normalisation'] and not followed_by_softmax:
        scale_add = 0.0
        scale_mul = 1.0
        params[pp(prefix,'ln_b')] = scale_add * numpy.ones((1*nout)).astype(floatX)
        params[pp(prefix,'ln_s')] = scale_mul * numpy.ones((1*nout)).astype(floatX)

    if options['weight_normalisation'] and not followed_by_softmax:
        scale_mul = 1.0
        params[pp(prefix,'W_wns')] = scale_mul * numpy.ones((1*nout)).astype(floatX)

    return params


def fflayer(tparams, state_below, options, dropout, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', W=None, b=None, dropout_probability=0, followed_by_softmax=False, **kwargs):
    if W == None:
        W = tparams[pp(prefix, 'W')]
    if b == None:
        b = tparams[pp(prefix, 'b')]

    # for three-dimensional tensors, we assume that first dimension is number of timesteps
    # we want to apply same mask to all timesteps
    if state_below.ndim == 3:
        dropout_shape = (state_below.shape[1], state_below.shape[2])
    else:
        dropout_shape = state_below.shape
    dropout_mask = dropout(dropout_shape, dropout_probability)

    if options['weight_normalisation'] and not followed_by_softmax:
         W = weight_norm(W, tparams[pp(prefix, 'W_wns')])
    preact = tensor.dot(state_below*dropout_mask, W) + b

    if options['layer_normalisation'] and not followed_by_softmax:
        preact = layer_norm(preact, tparams[pp(prefix,'ln_b')], tparams[pp(prefix,'ln_s')])

    return eval(activ)(preact)

# embedding layer
def param_init_embedding_layer(options, params, n_words, dims, factors=None, prefix='', suffix=''):
    if factors == None:
        factors = 1
        dims = [dims]
    for factor in xrange(factors):
        params[prefix+embedding_name(factor)+suffix] = norm_weight(n_words, dims[factor])
    return params

def embedding_layer(tparams, ids, factors=None, prefix='', suffix=''):
    do_reshape = False
    if factors == None:
        if ids.ndim > 1:
            do_reshape = True
            n_timesteps = ids.shape[0]
            n_samples = ids.shape[1]
        emb = tparams[prefix+embedding_name(0)+suffix][ids.flatten()]
    else:
        if ids.ndim > 2:
          do_reshape = True
          n_timesteps = ids.shape[1]
          n_samples = ids.shape[2]
        emb_list = [tparams[prefix+embedding_name(factor)+suffix][ids[factor].flatten()] for factor in xrange(factors)]
        emb = concatenate(emb_list, axis=1)
    if do_reshape:
        emb = emb.reshape((n_timesteps, n_samples, -1))

    return emb

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None,
                   recurrence_transition_depth=1,
                   **kwargs):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    scale_add = 0.0
    scale_mul = 1.0

    for i in xrange(recurrence_transition_depth):
        suffix = '' if i == 0 else ('_drt_%s' % i)
        # recurrent transformation weights for gates
        params[pp(prefix, 'b'+suffix)] = numpy.zeros((2 * dim,)).astype(floatX)
        U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
        params[pp(prefix, 'U'+suffix)] = U
        # recurrent transformation weights for hidden state proposal
        params[pp(prefix, 'bx'+suffix)] = numpy.zeros((dim,)).astype(floatX)
        Ux = ortho_weight(dim)
        params[pp(prefix, 'Ux'+suffix)] = Ux
        if options['layer_normalisation']:
            params[pp(prefix,'U%s_lnb' % suffix)] = scale_add * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'U%s_lns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        if options['weight_normalisation']:
            params[pp(prefix,'U%s_wns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_wns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)

        if i == 0:
            # embedding to gates transformation weights, biases
            W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
            params[pp(prefix, 'W'+suffix)] = W
            # embedding to hidden state proposal weights, biases
            Wx = norm_weight(nin, dim)
            params[pp(prefix, 'Wx'+suffix)] = Wx
            if options['layer_normalisation']:
                params[pp(prefix,'W%s_lnb' % suffix)] = scale_add * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'W%s_lns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix,'W%s_wns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_wns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)

    return params


def gru_layer(tparams, state_below, options, dropout, prefix='gru',
              mask=None, one_step=False,
              init_state=None,
              dropout_probability_below=0,
              dropout_probability_rec=0,
              recurrence_transition_depth=1,
              truncate_gradient=-1,
              profile=False,
              **kwargs):

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    dim = tparams[pp(prefix, 'Ux')].shape[1]

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']: 
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim))

    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    below_dropout = dropout((n_samples, dim_below), dropout_probability_below, num=2)
    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=2*(recurrence_transition_depth))

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_list, state_belowx_list = [], []

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'W'))) + tparams[pp(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'Wx'))) + tparams[pp(prefix, 'bx')]
    if options['layer_normalisation']:
        state_below_ = layer_norm(state_below_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
        state_belowx = layer_norm(state_belowx, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])
    state_below_list.append(state_below_)
    state_belowx_list.append(state_belowx)

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(*args):
        n_ins = 1
        m_ = args[0]
        x_list = args[1:1+n_ins]
        xx_list = args[1+n_ins:1+2*n_ins]
        h_, rec_dropout = args[-2], args[-1]

        h_prev = h_
        for i in xrange(recurrence_transition_depth):
            suffix = '' if i == 0 else ('_drt_%s' % i)
            if i == 0:
                x_cur = x_list[i]
                xx_cur = xx_list[i]
            else:
                x_cur = tparams[pp(prefix, 'b'+suffix)]
                xx_cur = tparams[pp(prefix, 'bx'+suffix)]

            preact = tensor.dot(h_prev*rec_dropout[0+2*i], wn(pp(prefix, 'U'+suffix)))
            if options['layer_normalisation']:
                preact = layer_norm(preact, tparams[pp(prefix, 'U%s_lnb' % suffix)], tparams[pp(prefix, 'U%s_lns' % suffix)])
            preact += x_cur

            # reset and update gates
            r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

            # compute the hidden state proposal
            preactx = tensor.dot(h_prev*rec_dropout[1+2*i], wn(pp(prefix, 'Ux'+suffix)))
            if options['layer_normalisation']:
                preactx = layer_norm(preactx, tparams[pp(prefix, 'Ux%s_lnb' % suffix)], tparams[pp(prefix, 'Ux%s_lns' % suffix)])
            preactx = preactx * r
            preactx = preactx + xx_cur

            # hidden state proposal
            h = tensor.tanh(preactx)

            # leaky integrate and obtain next hidden state
            h = u * h_prev + (1. - u) * h
            h = m_[:, None] * h + (1. - m_)[:, None] * h_prev
            h_prev = h

        return h

    # prepare scan arguments
    seqs = [mask] + state_below_list + state_belowx_list
    _step = _step_slice
    shared_vars = [rec_dropout]

    if one_step:
        rval = _step(*(seqs + [init_state] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_state,
                                non_sequences=shared_vars,
                                name=pp(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=truncate_gradient,
                                profile=profile,
                                strict=False)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None,
                        recurrence_transition_depth=2):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    scale_add = 0.0
    scale_mul = 1.0

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype(floatX)
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    for i in xrange(recurrence_transition_depth - 1):
        suffix = '' if i == 0 else ('_drt_%s' % i)
        U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
        params[pp(prefix, 'U_nl'+suffix)] = U_nl
        params[pp(prefix, 'b_nl'+suffix)] = numpy.zeros((2 * dim_nonlin,)).astype(floatX)
        Ux_nl = ortho_weight(dim_nonlin)
        params[pp(prefix, 'Ux_nl'+suffix)] = Ux_nl
        params[pp(prefix, 'bx_nl'+suffix)] = numpy.zeros((dim_nonlin,)).astype(floatX)
        
        if options['layer_normalisation']:
            params[pp(prefix,'U_nl%s_lnb' % suffix)] = scale_add * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'U_nl%s_lns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        if options['weight_normalisation']:
            params[pp(prefix,'U_nl%s_wns') % suffix] = scale_mul * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_wns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)

        # context to LSTM
        if i == 0:
            Wc = norm_weight(dimctx, dim*2)
            params[pp(prefix, 'Wc'+suffix)] = Wc
            Wcx = norm_weight(dimctx, dim)
            params[pp(prefix, 'Wcx'+suffix)] = Wcx
            if options['layer_normalisation']:
                params[pp(prefix,'Wc%s_lnb') % suffix] = scale_add * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wc%s_lns') % suffix] = scale_mul * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_lnb') % suffix] = scale_add * numpy.ones((1*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_lns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix,'Wc%s_wns') % suffix] = scale_mul * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_wns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)          

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att

    if options['layer_normalisation']:
        # layer-normalization parameters
        params[pp(prefix,'W_lnb')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'W_lns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'U_lnb')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'U_lns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'Wx_lnb')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Wx_lns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_lnb')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_lns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'W_comb_att_lnb')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'W_comb_att_lns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_lnb')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_lns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    if options['weight_normalisation']:
        params[pp(prefix,'W_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'U_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'Wx_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'W_comb_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'U_att_wns')] = scale_mul * numpy.ones((1*1)).astype(floatX)

    return params


def gru_cond_layer(tparams, state_below, options, dropout, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   pctx_=None,
                   recurrence_transition_depth=2,
                   truncate_gradient=-1,
                   profile=False,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num= 1 + 2 * recurrence_transition_depth)
    
    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']:
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    below_dropout = dropout((n_samples, dim_below),  dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2*options['dim']), dropout_probability_ctx, num=4)

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim))

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    if pctx_ is None:
        pctx_ = tensor.dot(context*ctx_dropout[0], wn(pp(prefix, 'Wc_att'))) +\
            tparams[pp(prefix, 'b_att')]

    if options['layer_normalisation']:
        pctx_ = layer_norm(pctx_, tparams[pp(prefix,'Wc_att_lnb')], tparams[pp(prefix,'Wc_att_lns')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the previous output word embedding
    state_belowx = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'Wx'))) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'W'))) +\
        tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout):
    # def _step_slice(m_, x_, xx_, h_, ctx_, pctx_, cc_, rec_dropout, ctx_dropout):
        if options['layer_normalisation']:
            x_ = layer_norm(x_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
            xx_ = layer_norm(xx_, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])

        preact1 = tensor.dot(h_*rec_dropout[0], wn(pp(prefix, 'U')))
        if options['layer_normalisation']:
            preact1 = layer_norm(preact1, tparams[pp(prefix, 'U_lnb')], tparams[pp(prefix, 'U_lns')])
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_*rec_dropout[1], wn(pp(prefix, 'Ux')))
        if options['layer_normalisation']:
            preactx1 = layer_norm(preactx1, tparams[pp(prefix, 'Ux_lnb')], tparams[pp(prefix, 'Ux_lns')])
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1*rec_dropout[2], wn(pp(prefix, 'W_comb_att')))
        if options['layer_normalisation']:
            pstate_ = layer_norm(pstate_, tparams[pp(prefix, 'W_comb_att_lnb')], tparams[pp(prefix, 'W_comb_att_lns')])
        pctx__ = pctx_ + pstate_[None, :, :]

        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__*ctx_dropout[1], wn(pp(prefix, 'U_att')))+tparams[pp(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        h2_prev = h1
        for i in xrange(recurrence_transition_depth - 1):
            suffix = '' if i == 0 else ('_drt_%s' % i)

            preact2 = tensor.dot(h2_prev*rec_dropout[3+2*i], wn(pp(prefix, 'U_nl'+suffix)))+tparams[pp(prefix, 'b_nl'+suffix)]
            if options['layer_normalisation']:
                preact2 = layer_norm(preact2, tparams[pp(prefix, 'U_nl%s_lnb' % suffix)], tparams[pp(prefix, 'U_nl%s_lns' % suffix)])
            if i == 0:
                ctx1_ = tensor.dot(ctx_*ctx_dropout[2], wn(pp(prefix, 'Wc'+suffix))) # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx1_ = layer_norm(ctx1_, tparams[pp(prefix, 'Wc%s_lnb' % suffix)], tparams[pp(prefix, 'Wc%s_lns' % suffix)])
                preact2 += ctx1_
            preact2 = tensor.nnet.sigmoid(preact2)

            r2 = _slice(preact2, 0, dim)
            u2 = _slice(preact2, 1, dim)

            preactx2 = tensor.dot(h2_prev*rec_dropout[4+2*i], wn(pp(prefix, 'Ux_nl'+suffix)))+tparams[pp(prefix, 'bx_nl'+suffix)]
            if options['layer_normalisation']:
               preactx2 = layer_norm(preactx2, tparams[pp(prefix, 'Ux_nl%s_lnb' % suffix)], tparams[pp(prefix, 'Ux_nl%s_lns' % suffix)])
            preactx2 *= r2
            if i == 0:
               ctx2_ = tensor.dot(ctx_*ctx_dropout[3], wn(pp(prefix, 'Wcx'+suffix))) # dropout mask is shared over mini-steps
               if options['layer_normalisation']:
                   ctx2_ = layer_norm(ctx2_, tparams[pp(prefix, 'Wcx%s_lnb' % suffix)], tparams[pp(prefix, 'Wcx%s_lns' % suffix)])
               preactx2 += ctx2_
            h2 = tensor.tanh(preactx2)

            h2 = u2 * h2_prev + (1. - u2) * h2
            h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h2_prev
            h2_prev = h2

        return h2, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = []

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.zeros((n_samples,
                                                               context.shape[2])),
                                                  tensor.zeros((n_samples,
                                                               context.shape[0]))
                                                  ],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_gradient,
                                    profile=profile,
                                    strict=False)
    return rval

def param_init_gru_cond_ctx_nogate(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None,
                        recurrence_transition_depth=2):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    scale_add = 0.0
    scale_mul = 1.0

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype(floatX)
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    for i in xrange(recurrence_transition_depth - 1):
        suffix = '' if i == 0 else ('_drt_%s' % i)
        U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                                  ortho_weight(dim_nonlin)], axis=1)
        params[pp(prefix, 'U_nl' + suffix)] = U_nl
        params[pp(prefix, 'b_nl' + suffix)] = numpy.zeros((2 * dim_nonlin,)).astype(floatX)
        Ux_nl = ortho_weight(dim_nonlin)
        params[pp(prefix, 'Ux_nl' + suffix)] = Ux_nl
        params[pp(prefix, 'bx_nl' + suffix)] = numpy.zeros((dim_nonlin,)).astype(floatX)

        params[pp(prefix, 'W_hist' + suffix)] = norm_weight(nin, dim * 2).astype(floatX)
        params[pp(prefix, 'Wx_hist' + suffix)] = norm_weight(nin, dim).astype(floatX)
        params[pp(prefix, 'b_hist' + suffix)] = numpy.zeros((dim * 2,)).astype(floatX)
        params[pp(prefix, 'bx_hist' + suffix)] = numpy.zeros((dim,)).astype(floatX)

        if options['layer_normalisation']:
            params[pp(prefix, 'U_nl%s_lnb' % suffix)] = scale_add * numpy.ones((2 * dim)).astype(floatX)
            params[pp(prefix, 'U_nl%s_lns' % suffix)] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
            params[pp(prefix, 'Ux_nl%s_lnb' % suffix)] = scale_add * numpy.ones((1 * dim)).astype(floatX)
            params[pp(prefix, 'Ux_nl%s_lns' % suffix)] = scale_mul * numpy.ones((1 * dim)).astype(floatX)

        if options['weight_normalisation']:
            params[pp(prefix, 'U_nl%s_wns') % suffix] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
            params[pp(prefix, 'Ux_nl%s_wns') % suffix] = scale_mul * numpy.ones((1 * dim)).astype(floatX)


            params[pp(prefix, 'W_hist_%swns') % suffix] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
            params[pp(prefix, 'Wx_hist_%swns') % suffix] = scale_mul * numpy.ones((1 * dim)).astype(floatX)

        # context to LSTM
        if i == 0:
            Wc = norm_weight(dimctx, dim * 2)
            params[pp(prefix, 'Wc' + suffix)] = Wc
            Wcx = norm_weight(dimctx, dim)
            params[pp(prefix, 'Wcx' + suffix)] = Wcx
            if options['layer_normalisation']:
                params[pp(prefix, 'Wc%s_lnb') % suffix] = scale_add * numpy.ones((2 * dim)).astype(floatX)
                params[pp(prefix, 'Wc%s_lns') % suffix] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
                params[pp(prefix, 'Wcx%s_lnb') % suffix] = scale_add * numpy.ones((1 * dim)).astype(floatX)
                params[pp(prefix, 'Wcx%s_lns') % suffix] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix, 'Wc%s_wns') % suffix] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
                params[pp(prefix, 'Wcx%s_wns') % suffix] = scale_mul * numpy.ones((1 * dim)).astype(floatX)

                # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    params[pp(prefix, 'Wctx_comb_att')] = norm_weight(dim, nin)
    params[pp(prefix, 'W_pctx_att')] = norm_weight(2 * dim, nin)

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att

    params[pp(prefix, 'Uctx_att')] = norm_weight(nin, 1)
    params[pp(prefix, 'cctx_tt')] = numpy.zeros((1,)).astype(floatX)

    if options['layer_normalisation']:
        # layer-normalization parameters
        params[pp(prefix, 'W_lnb')] = scale_add * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'W_lns')] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'U_lnb')] = scale_add * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'U_lns')] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'Wx_lnb')] = scale_add * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'Wx_lns')] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'Ux_lnb')] = scale_add * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'Ux_lns')] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'W_comb_att_lnb')] = scale_add * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'W_comb_att_lns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'Wc_att_lnb')] = scale_add * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'Wc_att_lns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'Wctx_comb_att_lnb')] = scale_add * numpy.ones((1 * nin)).astype(floatX)
        params[pp(prefix, 'Wctx_comb_att_lns')] = scale_mul * numpy.ones((1 * nin)).astype(floatX)
        params[pp(prefix, 'W_pctx_att_lnb')] = scale_add * numpy.ones((1 * nin)).astype(floatX)
        params[pp(prefix, 'W_pctx_att_lns')] = scale_mul * numpy.ones((1 * nin)).astype(floatX)
    if options['weight_normalisation']:
        params[pp(prefix, 'W_wns')] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'U_wns')] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'Wx_wns')] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'Ux_wns')] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'W_comb_att_wns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'Wc_att_wns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'U_att_wns')] = scale_mul * numpy.ones((1 * 1)).astype(floatX)
        params[pp(prefix, 'Wctx_comb_att_wns')] = scale_mul * numpy.ones((1 * nin)).astype(floatX)
        params[pp(prefix, 'W_pctx_att_wns')] = scale_mul * numpy.ones((1 * nin)).astype(floatX)

        params[pp(prefix, 'Uctx_att_wns')] = scale_mul * numpy.ones((1 * 1)).astype(floatX)


    return params

def gru_cond_layer_ctx_nogate(tparams, state_below, options, dropout, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   pctx_=None,
                   recurrence_transition_depth=2,
                   truncate_gradient=-1,
                   emb_pctx=None,
                   pctx_mask=None,
                   profile=False,
                   **kwargs):
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=1 + 2 * recurrence_transition_depth)

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']:
            return weight_norm(param, tparams[param_name + '_wns'])
        else:
            return param

    below_dropout = dropout((n_samples, dim_below), dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2 * options['dim']), dropout_probability_ctx, num=4)
    pctx_dropout = dropout((n_samples, options['dim_word']), dropout_probability_below, num=1)


    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim))

    if emb_pctx is None:
        emb_pctx = tensor.alloc(0., 1, 1, 1)

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    if pctx_ is None:
        pctx_ = tensor.dot(context * ctx_dropout[0], wn(pp(prefix, 'Wc_att'))) + \
                tparams[pp(prefix, 'b_att')]

    if options['layer_normalisation']:
        pctx_ = layer_norm(pctx_, tparams[pp(prefix, 'Wc_att_lnb')], tparams[pp(prefix, 'Wc_att_lns')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is the previous output word embedding
    state_belowx = tensor.dot(state_below * below_dropout[0], wn(pp(prefix, 'Wx'))) + \
                   tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below * below_dropout[1], wn(pp(prefix, 'W'))) + \
                   tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout, pctx_dropout, emb_pctx):

        if options['layer_normalisation']:
            x_ = layer_norm(x_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
            xx_ = layer_norm(xx_, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])

        preact1 = tensor.dot(h_ * rec_dropout[0], wn(pp(prefix, 'U')))
        if options['layer_normalisation']:
            preact1 = layer_norm(preact1, tparams[pp(prefix, 'U_lnb')], tparams[pp(prefix, 'U_lns')])
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_ * rec_dropout[1], wn(pp(prefix, 'Ux')))
        if options['layer_normalisation']:
            preactx1 = layer_norm(preactx1, tparams[pp(prefix, 'Ux_lnb')], tparams[pp(prefix, 'Ux_lns')])
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1 * rec_dropout[2], wn(pp(prefix, 'W_comb_att')))
        if options['layer_normalisation']:
            pstate_ = layer_norm(pstate_, tparams[pp(prefix, 'W_comb_att_lnb')], tparams[pp(prefix, 'W_comb_att_lns')])
        pctx__ = pctx_ + pstate_[None, :, :]
        # pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__ * ctx_dropout[1], wn(pp(prefix, 'U_att'))) + tparams[pp(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        h2_prev = h1
        for i in xrange(recurrence_transition_depth - 1):
            suffix = '' if i == 0 else ('_drt_%s' % i)

            pctxstate = tensor.dot(h2_prev * rec_dropout[2], wn(pp(prefix, 'Wctx_comb_att')))
            if options['layer_normalisation']:
                pctxstate = layer_norm(pctxstate, tparams[pp(prefix, 'Wctx_comb_att_lnb')],
                                       tparams[pp(prefix, 'Wctx_comb_att_lns')])
            ppctx_ = tensor.dot(ctx_, wn(pp(prefix, 'W_pctx_att')))
            if options['layer_normalisation']:
                ppctx_ = layer_norm(ppctx_, tparams[pp(prefix, 'W_pctx_att_lnb')],
                                    tparams[pp(prefix, 'W_pctx_att_lns')])
            pctxctx__ = emb_pctx + pctxstate[None, :, :] + ppctx_[None, :, :]
            pctxctx__ = tensor.tanh(pctxctx__)
            beta = tensor.dot(pctxctx__ * pctx_dropout, wn(pp(prefix, 'Uctx_att'))) + tparams[pp(prefix, 'cctx_tt')]
            beta = beta.reshape([beta.shape[0] * beta.shape[1], beta.shape[2]])
            beta = tensor.exp(beta - beta.max(0, keepdims=True))
            if pctx_mask:
                beta = beta * pctx_mask.reshape([pctx_mask.shape[0] * pctx_mask.shape[1], pctx_mask.shape[2]])
            beta = beta / beta.sum(0, keepdims=True)

            pcsctx = emb_pctx.reshape([emb_pctx.shape[0] * emb_pctx.shape[1], emb_pctx.shape[2], -1]) * beta[:, :, None]
            pcsctx = pcsctx.sum(0)

            preact2 = tensor.dot(h2_prev * rec_dropout[3 + 2 * i], wn(pp(prefix, 'U_nl' + suffix))) + tparams[
                pp(prefix, 'b_nl' + suffix)]
            if options['layer_normalisation']:
                preact2 = layer_norm(preact2, tparams[pp(prefix, 'U_nl%s_lnb' % suffix)],
                                     tparams[pp(prefix, 'U_nl%s_lns' % suffix)])
            if i == 0:
                ctx1_ = tensor.dot(ctx_ * ctx_dropout[2],
                                   wn(pp(prefix, 'Wc' + suffix)))  # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx1_ = layer_norm(ctx1_, tparams[pp(prefix, 'Wc%s_lnb' % suffix)],
                                       tparams[pp(prefix, 'Wc%s_lns' % suffix)])
                preact2 += ctx1_

            preact2 += tensor.dot(pcsctx, wn(pp(prefix, 'W_hist'))) + tparams[pp(prefix, 'b_hist')]

            r2 = _slice(preact2, 0, dim)
            u2 = _slice(preact2, 1, dim)

            preactx2 = tensor.dot(h2_prev * rec_dropout[4 + 2 * i], wn(pp(prefix, 'Ux_nl' + suffix))) + tparams[
                pp(prefix, 'bx_nl' + suffix)]
            if options['layer_normalisation']:
                preactx2 = layer_norm(preactx2, tparams[pp(prefix, 'Ux_nl%s_lnb' % suffix)],
                                      tparams[pp(prefix, 'Ux_nl%s_lns' % suffix)])
            preactx2 *= r2
            if i == 0:
                ctx2_ = tensor.dot(ctx_ * ctx_dropout[3],
                                   wn(pp(prefix, 'Wcx' + suffix)))  # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx2_ = layer_norm(ctx2_, tparams[pp(prefix, 'Wcx%s_lnb' % suffix)],
                                       tparams[pp(prefix, 'Wcx%s_lns' % suffix)])
                preactx2 += ctx2_

            preactx2 += tensor.dot(pcsctx, wn(pp(prefix, 'Wx_hist'))) + tparams[pp(prefix, 'bx_hist')]

            h2 = tensor.tanh(preactx2)

            h2 = u2 * h2_prev + (1. - u2) * h2
            h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h2_prev
            h2_prev = h2

        return h2, ctx_, alpha.T # , beta.T, z  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = []

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout, pctx_dropout, emb_pctx] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.zeros((n_samples,
                                                                context.shape[2])),
                                                  tensor.zeros((n_samples,
                                                                context.shape[0]))],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout, pctx_dropout, emb_pctx] + shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_gradient,
                                    profile=profile,
                                    strict=False)

    if options['ctx_model']:
        return rval
    return rval[:3]

def param_init_gru_cond_ctx(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None,
                        recurrence_transition_depth=2):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    scale_add = 0.0
    scale_mul = 1.0

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype(floatX)
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    for i in xrange(recurrence_transition_depth - 1):
        suffix = '' if i == 0 else ('_drt_%s' % i)
        U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                                  ortho_weight(dim_nonlin)], axis=1)
        params[pp(prefix, 'U_nl' + suffix)] = U_nl
        params[pp(prefix, 'b_nl' + suffix)] = numpy.zeros((2 * dim_nonlin,)).astype(floatX)
        Ux_nl = ortho_weight(dim_nonlin)
        params[pp(prefix, 'Ux_nl' + suffix)] = Ux_nl
        params[pp(prefix, 'bx_nl' + suffix)] = numpy.zeros((dim_nonlin,)).astype(floatX)

        params[pp(prefix, 'Uz' + suffix)] = norm_weight(dim, dim * 3).astype(floatX)
        params[pp(prefix, 'Cz' + suffix)] = norm_weight(dim * 2, dim * 3).astype(floatX)
        params[pp(prefix, 'b_z' + suffix)] = numpy.zeros((dim * 3,)).astype(floatX)

        if options['ctx_sep_enc']:
            params[pp(prefix, 'W_hier' + suffix)] = norm_weight(nin, nin).astype(floatX)
            params[pp(prefix, 'b_hier' + suffix)] = numpy.zeros(nin).astype(floatX)

        if options['ctx_rnn']:
            params[pp(prefix, 'CCz' + suffix)] = norm_weight(dimctx, dim * 3).astype(floatX)
        else:
            params[pp(prefix, 'CCz' + suffix)] = norm_weight(nin, dim * 3).astype(floatX)

        if options['ctx_rnn']:
            params[pp(prefix, 'W_hist' + suffix)] = norm_weight(dimctx, dim * 2).astype(floatX)
            params[pp(prefix, 'Wx_hist' + suffix)] = norm_weight(dimctx, dim).astype(floatX)
        else:
            params[pp(prefix, 'W_hist' + suffix)] = norm_weight(nin, dim * 2).astype(floatX)
            params[pp(prefix, 'Wx_hist' + suffix)] = norm_weight(nin, dim).astype(floatX)

        if options['layer_normalisation']:
            params[pp(prefix, 'U_nl%s_lnb' % suffix)] = scale_add * numpy.ones((2 * dim)).astype(floatX)
            params[pp(prefix, 'U_nl%s_lns' % suffix)] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
            params[pp(prefix, 'Ux_nl%s_lnb' % suffix)] = scale_add * numpy.ones((1 * dim)).astype(floatX)
            params[pp(prefix, 'Ux_nl%s_lns' % suffix)] = scale_mul * numpy.ones((1 * dim)).astype(floatX)


        if options['weight_normalisation']:
            params[pp(prefix, 'U_nl%s_wns') % suffix] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
            params[pp(prefix, 'Ux_nl%s_wns') % suffix] = scale_mul * numpy.ones((1 * dim)).astype(floatX)

            params[pp(prefix, 'Uz_%swns') % suffix] = scale_mul * numpy.ones((3 * dim)).astype(floatX)
            params[pp(prefix, 'Cz_%swns') % suffix] = scale_mul * numpy.ones((3 * dim)).astype(floatX)
            params[pp(prefix, 'CCz_%swns') % suffix] = scale_mul * numpy.ones((3 * dim)).astype(floatX)

            params[pp(prefix, 'W_hist_%swns') % suffix] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
            params[pp(prefix, 'Wx_hist_%swns') % suffix] = scale_mul * numpy.ones((1 * dim)).astype(floatX)

            if options['ctx_sep_enc']:
                params[pp(prefix, 'W_hier%s_wns') % suffix] = scale_mul * numpy.ones((nin)).astype(floatX)

        # context to LSTM
        if i == 0:
            Wc = norm_weight(dimctx, dim * 2)
            params[pp(prefix, 'Wc' + suffix)] = Wc
            Wcx = norm_weight(dimctx, dim)
            params[pp(prefix, 'Wcx' + suffix)] = Wcx
            if options['layer_normalisation']:
                params[pp(prefix, 'Wc%s_lnb') % suffix] = scale_add * numpy.ones((2 * dim)).astype(floatX)
                params[pp(prefix, 'Wc%s_lns') % suffix] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
                params[pp(prefix, 'Wcx%s_lnb') % suffix] = scale_add * numpy.ones((1 * dim)).astype(floatX)
                params[pp(prefix, 'Wcx%s_lns') % suffix] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix, 'Wc%s_wns') % suffix] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
                params[pp(prefix, 'Wcx%s_wns') % suffix] = scale_mul * numpy.ones((1 * dim)).astype(floatX)

                # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    if options['ctx_rnn']:
        params[pp(prefix, 'Wctx_comb_att')] = norm_weight(dim, dimctx)
        params[pp(prefix, 'W_pctx_att')] = norm_weight(2 * dim, dimctx)
    else:
        params[pp(prefix, 'Wctx_comb_att')] = norm_weight(dim, nin)
        params[pp(prefix, 'W_pctx_att')] = norm_weight(2 * dim, nin)

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att

    if options['ctx_rnn']:
        params[pp(prefix, 'Uctx_att')] = norm_weight(dimctx, 1)
    else:
        params[pp(prefix, 'Uctx_att')] = norm_weight(nin, 1)
    params[pp(prefix, 'cctx_tt')] = numpy.zeros((1,)).astype(floatX)

    if options['layer_normalisation']:
        # layer-normalization parameters
        params[pp(prefix, 'W_lnb')] = scale_add * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'W_lns')] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'U_lnb')] = scale_add * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'U_lns')] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'Wx_lnb')] = scale_add * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'Wx_lns')] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'Ux_lnb')] = scale_add * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'Ux_lns')] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'W_comb_att_lnb')] = scale_add * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'W_comb_att_lns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'Wc_att_lnb')] = scale_add * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'Wc_att_lns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)


        if options['ctx_rnn']:
            params[pp(prefix, 'Wctx_comb_att_lnb')] = scale_add * numpy.ones((1 * dimctx)).astype(floatX)
            params[pp(prefix, 'Wctx_comb_att_lns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
            params[pp(prefix, 'W_pctx_att_lnb')] = scale_add * numpy.ones((1 * dimctx)).astype(floatX)
            params[pp(prefix, 'W_pctx_att_lns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
        else:
            params[pp(prefix, 'Wctx_comb_att_lnb')] = scale_add * numpy.ones((1 * nin)).astype(floatX)
            params[pp(prefix, 'Wctx_comb_att_lns')] = scale_mul * numpy.ones((1 * nin)).astype(floatX)
            params[pp(prefix, 'W_pctx_att_lnb')] = scale_add * numpy.ones((1 * nin)).astype(floatX)
            params[pp(prefix, 'W_pctx_att_lns')] = scale_mul * numpy.ones((1 * nin)).astype(floatX)


    if options['weight_normalisation']:
        params[pp(prefix, 'W_wns')] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'U_wns')] = scale_mul * numpy.ones((2 * dim)).astype(floatX)
        params[pp(prefix, 'Wx_wns')] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'Ux_wns')] = scale_mul * numpy.ones((1 * dim)).astype(floatX)
        params[pp(prefix, 'W_comb_att_wns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'Wc_att_wns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
        params[pp(prefix, 'U_att_wns')] = scale_mul * numpy.ones((1 * 1)).astype(floatX)

        if options['ctx_rnn']:
            params[pp(prefix, 'Wctx_comb_att_wns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
            params[pp(prefix, 'W_pctx_att_wns')] = scale_mul * numpy.ones((1 * dimctx)).astype(floatX)
        else:
            params[pp(prefix, 'Wctx_comb_att_wns')] = scale_mul * numpy.ones((1 * nin)).astype(floatX)
            params[pp(prefix, 'W_pctx_att_wns')] = scale_mul * numpy.ones((1 * nin)).astype(floatX)

        params[pp(prefix, 'Uctx_att_wns')] = scale_mul * numpy.ones((1 * 1)).astype(floatX)


    return params


def gru_cond_layer_ctx(tparams, state_below, options, dropout, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   pctx_=None,
                   recurrence_transition_depth=2,
                   truncate_gradient=-1,
                   emb_pctx=None,
                   pctx_mask=None,
                   profile=False,
                   **kwargs):
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=1 + 2 * recurrence_transition_depth)

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']:
            return weight_norm(param, tparams[param_name + '_wns'])
        else:
            return param

    below_dropout = dropout((n_samples, dim_below), dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2 * options['dim']), dropout_probability_ctx, num=4)
    if options['ctx_rnn']:
        pctx_dropout = dropout((n_samples, 2 * options['dim']), dropout_probability_below, num=1)
    else:
        pctx_dropout = dropout((n_samples, options['dim_word']), dropout_probability_below, num=1)

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim))

    if emb_pctx is None:
        emb_pctx = tensor.alloc(0., 1, 1, 1)

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    if pctx_ is None:
        pctx_ = tensor.dot(context * ctx_dropout[0], wn(pp(prefix, 'Wc_att'))) + \
                tparams[pp(prefix, 'b_att')]

    if options['layer_normalisation']:
        pctx_ = layer_norm(pctx_, tparams[pp(prefix, 'Wc_att_lnb')], tparams[pp(prefix, 'Wc_att_lns')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is the previous output word embedding
    state_belowx = tensor.dot(state_below * below_dropout[0], wn(pp(prefix, 'Wx'))) + \
                   tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below * below_dropout[1], wn(pp(prefix, 'W'))) + \
                   tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, beta_, z_, pctx_, cc_, rec_dropout, ctx_dropout, pctx_dropout, emb_pctx):

        beta, z = tensor.alloc(0., n_samples, emb_pctx.shape[1]), tensor.alloc(0., n_samples, 3 * options['dim'])

        if options['layer_normalisation']:
            x_ = layer_norm(x_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
            xx_ = layer_norm(xx_, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])

        preact1 = tensor.dot(h_ * rec_dropout[0], wn(pp(prefix, 'U')))
        if options['layer_normalisation']:
            preact1 = layer_norm(preact1, tparams[pp(prefix, 'U_lnb')], tparams[pp(prefix, 'U_lns')])
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_ * rec_dropout[1], wn(pp(prefix, 'Ux')))
        if options['layer_normalisation']:
            preactx1 = layer_norm(preactx1, tparams[pp(prefix, 'Ux_lnb')], tparams[pp(prefix, 'Ux_lns')])
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1 * rec_dropout[2], wn(pp(prefix, 'W_comb_att')))
        if options['layer_normalisation']:
            pstate_ = layer_norm(pstate_, tparams[pp(prefix, 'W_comb_att_lnb')], tparams[pp(prefix, 'W_comb_att_lns')])
        pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__ * ctx_dropout[1], wn(pp(prefix, 'U_att'))) + tparams[pp(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        h2_prev = h1
        for i in xrange(recurrence_transition_depth - 1):
            suffix = '' if i == 0 else ('_drt_%s' % i)

            pctxstate = tensor.dot(h2_prev * rec_dropout[2], wn(pp(prefix, 'Wctx_comb_att')))
            if options['layer_normalisation']:
                pctxstate = layer_norm(pctxstate, tparams[pp(prefix, 'Wctx_comb_att_lnb')],
                                       tparams[pp(prefix, 'Wctx_comb_att_lns')])
            ppctx_ = tensor.dot(ctx_, wn(pp(prefix, 'W_pctx_att')))
            if options['layer_normalisation']:
                ppctx_ = layer_norm(ppctx_, tparams[pp(prefix, 'W_pctx_att_lnb')],
                                    tparams[pp(prefix, 'W_pctx_att_lns')])

            pctxctx__ = emb_pctx + pctxstate[None, :, :] + ppctx_[None, :, :]
            pctxctx__ = tensor.tanh(pctxctx__)
            beta = tensor.dot(pctxctx__ * pctx_dropout, wn(pp(prefix, 'Uctx_att'))) + tparams[pp(prefix, 'cctx_tt')]

            if options['ctx_rnn']:
                beta = beta.reshape([beta.shape[0], beta.shape[1]])
            else:
                beta = beta.reshape([beta.shape[0] * beta.shape[1], beta.shape[2]])
            beta = tensor.exp(beta - beta.max(0, keepdims=True))
            if pctx_mask:
                if options['ctx_rnn']:
                    beta = beta * pctx_mask
                else:
                    beta = beta * pctx_mask.reshape([pctx_mask.shape[0] * pctx_mask.shape[1], pctx_mask.shape[2]])
            beta = beta / beta.sum(0, keepdims=True)

            if options['ctx_rnn']:
                pcsctx = emb_pctx * beta[:, :, None]
            else:
                pcsctx = emb_pctx.reshape([emb_pctx.shape[0] * emb_pctx.shape[1], emb_pctx.shape[2], -1]) * beta[:, :, None]
            pcsctx = pcsctx.sum(0)

            z_ = tensor.dot(h2_prev, wn(pp(prefix, 'Uz' + suffix)))
            z_ += tensor.dot(ctx_, wn(pp(prefix, 'Cz' + suffix)))
            z_ += tensor.dot(pcsctx, wn(pp(prefix, 'CCz' + suffix))) + tparams[prefix + '_b_z']
            z = tensor.nnet.sigmoid(z_)

            z1 = _slice(z, 0, 2 * options['dim'])
            z2 = _slice(z, 2, 1 * options['dim'])


            preact2 = tensor.dot(h2_prev * rec_dropout[3 + 2 * i], wn(pp(prefix, 'U_nl' + suffix))) + tparams[
                pp(prefix, 'b_nl' + suffix)]
            if options['layer_normalisation']:
                preact2 = layer_norm(preact2, tparams[pp(prefix, 'U_nl%s_lnb' % suffix)],
                                     tparams[pp(prefix, 'U_nl%s_lns' % suffix)])
            if i == 0:
                ctx1_ = tensor.dot(ctx_ * ctx_dropout[2],
                                   wn(pp(prefix, 'Wc' + suffix)))  # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx1_ = layer_norm(ctx1_, tparams[pp(prefix, 'Wc%s_lnb' % suffix)],
                                       tparams[pp(prefix, 'Wc%s_lns' % suffix)])
                preact2 += ctx1_

            preact2 = (1. - z1) * preact2
            preact2 += z1 * tensor.dot(pcsctx, wn(pp(prefix, 'W_hist'))) + tparams[pp(prefix, 'b_nl' + suffix)]
            preact2 = tensor.nnet.sigmoid(preact2)

            r2 = _slice(preact2, 0, dim)
            u2 = _slice(preact2, 1, dim)

            preactx2 = tensor.dot(h2_prev * rec_dropout[4 + 2 * i], wn(pp(prefix, 'Ux_nl' + suffix))) + tparams[
                pp(prefix, 'bx_nl' + suffix)]
            if options['layer_normalisation']:
                preactx2 = layer_norm(preactx2, tparams[pp(prefix, 'Ux_nl%s_lnb' % suffix)],
                                      tparams[pp(prefix, 'Ux_nl%s_lns' % suffix)])
            preactx2 *= r2
            if i == 0:
                ctx2_ = tensor.dot(ctx_ * ctx_dropout[3],
                                   wn(pp(prefix, 'Wcx' + suffix)))  # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx2_ = layer_norm(ctx2_, tparams[pp(prefix, 'Wcx%s_lnb' % suffix)],
                                       tparams[pp(prefix, 'Wcx%s_lns' % suffix)])
                preactx2 += ctx2_

            preactx2 = (1. - z2) * preactx2
            preactx2 += z2 * tensor.dot(pcsctx, wn(pp(prefix, 'Wx_hist'))) + tparams[pp(prefix, 'bx_nl' + suffix)]

            h2 = tensor.tanh(preactx2)

            h2 = u2 * h2_prev + (1. - u2) * h2
            h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h2_prev
            h2_prev = h2

        return h2, ctx_, alpha.T, beta.T, z  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = []

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, None, None, pctx_, context, rec_dropout, ctx_dropout, pctx_dropout, emb_pctx] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.zeros((n_samples,
                                                                context.shape[2])),
                                                  tensor.zeros((n_samples,
                                                                context.shape[0])), tensor.alloc(0., n_samples, emb_pctx.shape[1]), tensor.alloc(0., n_samples, 3 * options['dim'])],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout, pctx_dropout, emb_pctx] + shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_gradient,
                                    profile=profile,
                                    strict=False)

    if options['ctx_model']:
        return rval
    return rval[:3]

def gru_cond_layer_mul_ctx(tparams, state_below, options, dropout, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   pctx_=None,
                   recurrence_transition_depth=2,
                   truncate_gradient=-1,
                   emb_pctx=None,
                   num_ctx_sents=0,
                   pctx_mask=None,
                   profile=False,
                   **kwargs):
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=1 + 2 * recurrence_transition_depth)

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']:
            return weight_norm(param, tparams[param_name + '_wns'])
        else:
            return param

    below_dropout = dropout((n_samples, dim_below), dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2 * options['dim']), dropout_probability_ctx, num=4)
    if options['ctx_rnn']:
        pctx_dropout = dropout((n_samples, 2 * options['dim']), dropout_probability_below, num=1)
    else:
        pctx_dropout = dropout((n_samples, options['dim_word']), dropout_probability_below, num=1)

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim))

    if emb_pctx is None:
        emb_pctx = tensor.alloc(0., 1, 1, 1)

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    if pctx_ is None:
        pctx_ = tensor.dot(context * ctx_dropout[0], wn(pp(prefix, 'Wc_att'))) + \
                tparams[pp(prefix, 'b_att')]

    if options['layer_normalisation']:
        pctx_ = layer_norm(pctx_, tparams[pp(prefix, 'Wc_att_lnb')], tparams[pp(prefix, 'Wc_att_lns')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is the previous output word embedding
    state_belowx = tensor.dot(state_below * below_dropout[0], wn(pp(prefix, 'Wx'))) + \
                   tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below * below_dropout[1], wn(pp(prefix, 'W'))) + \
                   tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, beta_, z_, pctx_, cc_, rec_dropout, ctx_dropout, pctx_dropout, emb_pctx):

        beta, z = tensor.alloc(0., n_samples, options['num_ctx_sents'] * emb_pctx.shape[1]), tensor.alloc(0., n_samples, 3 * options['dim'])

        if options['layer_normalisation']:
            x_ = layer_norm(x_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
            xx_ = layer_norm(xx_, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])

        preact1 = tensor.dot(h_ * rec_dropout[0], wn(pp(prefix, 'U')))
        if options['layer_normalisation']:
            preact1 = layer_norm(preact1, tparams[pp(prefix, 'U_lnb')], tparams[pp(prefix, 'U_lns')])
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_ * rec_dropout[1], wn(pp(prefix, 'Ux')))
        if options['layer_normalisation']:
            preactx1 = layer_norm(preactx1, tparams[pp(prefix, 'Ux_lnb')], tparams[pp(prefix, 'Ux_lns')])
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1 * rec_dropout[2], wn(pp(prefix, 'W_comb_att')))
        if options['layer_normalisation']:
            pstate_ = layer_norm(pstate_, tparams[pp(prefix, 'W_comb_att_lnb')], tparams[pp(prefix, 'W_comb_att_lns')])
        pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__ * ctx_dropout[1], wn(pp(prefix, 'U_att'))) + tparams[pp(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        h2_prev = h1
        for i in xrange(recurrence_transition_depth - 1):
            suffix = '' if i == 0 else ('_drt_%s' % i)

            pctxstate = tensor.dot(h2_prev * rec_dropout[2], wn(pp(prefix, 'Wctx_comb_att')))
            if options['layer_normalisation']:
                pctxstate = layer_norm(pctxstate, tparams[pp(prefix, 'Wctx_comb_att_lnb')],
                                       tparams[pp(prefix, 'Wctx_comb_att_lns')])
            ppctx_ = tensor.dot(ctx_, wn(pp(prefix, 'W_pctx_att')))
            if options['layer_normalisation']:
                ppctx_ = layer_norm(ppctx_, tparams[pp(prefix, 'W_pctx_att_lnb')],
                                    tparams[pp(prefix, 'W_pctx_att_lns')])

            pctxctx__ = emb_pctx + pctxstate[None, :, :] + ppctx_[None, :, :]
            pctxctx__ = tensor.tanh(pctxctx__)
            beta = tensor.dot(pctxctx__ * pctx_dropout, wn(pp(prefix, 'Uctx_att'))) + tparams[pp(prefix, 'cctx_tt')]
            if options['ctx_rnn']:
                beta = beta.reshape([beta.shape[0], beta.shape[1]])
            else:
                beta = beta.reshape([beta.shape[0] * beta.shape[1], beta.shape[2]])
            for j in range(num_ctx_sents):
                beta[j] = tensor.exp(beta[j] - beta[j].max(0, keepdims=True))
            if pctx_mask:
                if options['ctx_rnn']:
                    beta = beta * pctx_mask
                else:
                    beta = beta * pctx_mask.reshape([pctx_mask.shape[0] * pctx_mask.shape[1], pctx_mask.shape[2]])
            for j in range(num_ctx_sents):
                beta[j] = beta[j] / beta[j].sum(0, keepdims=True)
            if options['ctx_rnn']:
                pcsctx = emb_pctx * beta[:, :, None]
            else:
                pcsctx = tensor.dot(emb_pctx.reshape([emb_pctx.shape[0] * emb_pctx.shape[1], emb_pctx.shape[2], -1]) * beta[:, :, None], wn(pp(prefix, 'W_hier'))) + tparams[pp(prefix, 'b_hier')]
            pcsctx = pcsctx.sum(0)

            z_ = tensor.dot(h2_prev, wn(pp(prefix, 'Uz' + suffix)))
            z_ += tensor.dot(ctx_, wn(pp(prefix, 'Cz' + suffix)))
            z_ += tensor.dot(pcsctx, wn(pp(prefix, 'CCz' + suffix))) + tparams[prefix + '_b_z']
            z = tensor.nnet.sigmoid(z_)

            z1 = _slice(z, 0, 2 * options['dim'])
            z2 = _slice(z, 2, 1 * options['dim'])


            preact2 = tensor.dot(h2_prev * rec_dropout[3 + 2 * i], wn(pp(prefix, 'U_nl' + suffix))) + tparams[
                pp(prefix, 'b_nl' + suffix)]
            if options['layer_normalisation']:
                preact2 = layer_norm(preact2, tparams[pp(prefix, 'U_nl%s_lnb' % suffix)],
                                     tparams[pp(prefix, 'U_nl%s_lns' % suffix)])
            if i == 0:
                ctx1_ = tensor.dot(ctx_ * ctx_dropout[2],
                                   wn(pp(prefix, 'Wc' + suffix)))  # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx1_ = layer_norm(ctx1_, tparams[pp(prefix, 'Wc%s_lnb' % suffix)],
                                       tparams[pp(prefix, 'Wc%s_lns' % suffix)])
                preact2 += ctx1_

            preact2 = (1. - z1) * preact2
            preact2 += z1 * tensor.dot(pcsctx, wn(pp(prefix, 'W_hist'))) + tparams[pp(prefix, 'b_nl' + suffix)]
            preact2 = tensor.nnet.sigmoid(preact2)

            r2 = _slice(preact2, 0, dim)
            u2 = _slice(preact2, 1, dim)

            preactx2 = tensor.dot(h2_prev * rec_dropout[4 + 2 * i], wn(pp(prefix, 'Ux_nl' + suffix))) + tparams[
                pp(prefix, 'bx_nl' + suffix)]
            if options['layer_normalisation']:
                preactx2 = layer_norm(preactx2, tparams[pp(prefix, 'Ux_nl%s_lnb' % suffix)],
                                      tparams[pp(prefix, 'Ux_nl%s_lns' % suffix)])
            preactx2 *= r2
            if i == 0:
                ctx2_ = tensor.dot(ctx_ * ctx_dropout[3],
                                   wn(pp(prefix, 'Wcx' + suffix)))  # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx2_ = layer_norm(ctx2_, tparams[pp(prefix, 'Wcx%s_lnb' % suffix)],
                                       tparams[pp(prefix, 'Wcx%s_lns' % suffix)])
                preactx2 += ctx2_

            preactx2 = (1. - z2) * preactx2
            preactx2 += z2 * tensor.dot(pcsctx, wn(pp(prefix, 'Wx_hist'))) + tparams[pp(prefix, 'bx_nl' + suffix)]

            h2 = tensor.tanh(preactx2)

            h2 = u2 * h2_prev + (1. - u2) * h2
            h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h2_prev
            h2_prev = h2

        return h2, ctx_, alpha.T, beta.T, z  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = []

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, None, None, pctx_, context, rec_dropout, ctx_dropout, pctx_dropout, emb_pctx] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.zeros((n_samples,
                                                                context.shape[2])),
                                                  tensor.zeros((n_samples,
                                                                context.shape[0])), tensor.alloc(0., n_samples, options['num_ctx_sents'] * emb_pctx.shape[1]), tensor.alloc(0., n_samples, 3 * options['dim'])],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout, pctx_dropout, emb_pctx] + shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_gradient,
                                    profile=profile,
                                    strict=False)

    if options['ctx_model']:
        return rval
    return rval[:3]

# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None,
                   recurrence_transition_depth=1,
                   **kwargs):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    scale_add = 0.0
    scale_mul = 1.0

    for i in xrange(recurrence_transition_depth):
        suffix = '' if i == 0 else ('_drt_%s' % i)

        # recurrent transformation weights for gates

        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)],
                               axis=1)

        params[pp(prefix, 'U'+suffix)] = U
        params[pp(prefix, 'b'+suffix)] = numpy.zeros((3 * dim,)).astype(floatX)

        # recurrent transformation weights for hidden state proposal
        Ux = ortho_weight(dim)
        params[pp(prefix, 'Ux'+suffix)] = Ux
        params[pp(prefix, 'bx'+suffix)] = numpy.zeros((dim,)).astype(floatX)

        if options['layer_normalisation']:
            params[pp(prefix,'U%s_lnb' % suffix)] = scale_add * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'U%s_lns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        if options['weight_normalisation']:
            params[pp(prefix,'U%s_wns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_wns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)

        if i == 0:
            # embedding to gates transformation weights
            W = numpy.concatenate([norm_weight(nin, dim),
                                   norm_weight(nin, dim),
                                   norm_weight(nin, dim)],
                                   axis=1)
            params[pp(prefix, 'W'+suffix)] = W
            # embedding to hidden state proposal weights
            Wx = norm_weight(nin, dim)
            params[pp(prefix, 'Wx'+suffix)] = Wx
            if options['layer_normalisation']:
                params[pp(prefix,'W%s_lnb' % suffix)] = scale_add * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'W%s_lns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix,'W%s_wns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_wns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)

    return params


def lstm_layer(tparams, state_below, options, dropout, prefix='lstm',
              mask=None, one_step=False,
              init_state=None,
              dropout_probability_below=0,
              dropout_probability_rec=0,
              recurrence_transition_depth=1,
              truncate_gradient=-1,
              profile=False,
              **kwargs):

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    dim = tparams[pp(prefix, 'Ux')].shape[1]

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']: 
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim*2))

    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    below_dropout = dropout((n_samples, dim_below), dropout_probability_below, num=2)
    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=2*(recurrence_transition_depth))

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_list, state_belowx_list = [], []

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'W'))) + tparams[pp(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'Wx'))) + tparams[pp(prefix, 'bx')]
    if options['layer_normalisation']:
        state_below_ = layer_norm(state_below_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
        state_belowx = layer_norm(state_belowx, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])
    state_below_list.append(state_below_)
    state_belowx_list.append(state_belowx)

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(*args):
        n_ins = 1
        m_ = args[0]
        x_list = args[1:1+n_ins]
        xx_list = args[1+n_ins:1+2*n_ins]
        h_, rec_dropout = args[-2], args[-1]

        h_prev = _slice(h_, 0, dim)
        c_prev = _slice(h_, 1, dim)

        for i in xrange(recurrence_transition_depth):
            suffix = '' if i == 0 else ('_drt_%s' % i)
            if i == 0:
                x_cur = x_list[i]
                xx_cur = xx_list[i]
            else:
                x_cur = tparams[pp(prefix, 'b'+suffix)]
                xx_cur = tparams[pp(prefix, 'bx'+suffix)]

            preact = tensor.dot(h_prev*rec_dropout[0+2*i], wn(pp(prefix, 'U'+suffix)))
            if options['layer_normalisation']:
                preact = layer_norm(preact, tparams[pp(prefix, 'U%s_lnb' % suffix)], tparams[pp(prefix, 'U%s_lns' % suffix)])
            preact += x_cur

            # gates
            gate_i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            gate_f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
            gate_o = tensor.nnet.sigmoid(_slice(preact, 2, dim))

            # compute the hidden state proposal
            preactx = tensor.dot(h_prev*rec_dropout[1+2*i], wn(pp(prefix, 'Ux'+suffix)))
            if options['layer_normalisation']:
                preactx = layer_norm(preactx, tparams[pp(prefix, 'Ux%s_lnb' % suffix)], tparams[pp(prefix, 'Ux%s_lns' % suffix)])
            preactx += xx_cur

            c = tensor.tanh(preactx)
            c = gate_f * c_prev + gate_i * c
            h = gate_o * tensor.tanh(c)

            # if state is masked, simply copy previous
            h = m_[:, None] * h + (1. - m_)[:, None] * h_prev
            c = m_[:, None] * c + (1. - m_)[:, None] * c_prev
            h_prev = h
            c_prev = c

        h = concatenate([h, c], axis=1)

        return h

    # prepare scan arguments
    seqs = [mask] + state_below_list + state_belowx_list
    _step = _step_slice
    shared_vars = [rec_dropout]

    if one_step:
        rval = _step(*(seqs + [init_state] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_state,
                                non_sequences=shared_vars,
                                name=pp(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=truncate_gradient,
                                profile=profile,
                                strict=False)
    rval = [rval]
    return rval

# Conditional LSTM layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None,
                        recurrence_transition_depth=2):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    scale_add = 0.0
    scale_mul = 1.0

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim)],
                           axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((3 * dim,)).astype(floatX)

    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)],
                           axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    for i in xrange(recurrence_transition_depth - 1):
        suffix = '' if i == 0 else ('_drt_%s' % i)
        U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                                  ortho_weight(dim_nonlin),
                                  ortho_weight(dim_nonlin)],
                                  axis=1)
        params[pp(prefix, 'U_nl'+suffix)] = U_nl
        params[pp(prefix, 'b_nl'+suffix)] = numpy.zeros((3 * dim_nonlin,)).astype(floatX)
        Ux_nl = ortho_weight(dim_nonlin)
        params[pp(prefix, 'Ux_nl'+suffix)] = Ux_nl
        params[pp(prefix, 'bx_nl'+suffix)] = numpy.zeros((dim_nonlin,)).astype(floatX)
        
        if options['layer_normalisation']:
            params[pp(prefix,'U_nl%s_lnb' % suffix)] = scale_add * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'U_nl%s_lns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        if options['weight_normalisation']:
            params[pp(prefix,'U_nl%s_wns') % suffix] = scale_mul * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_wns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)

        # context to LSTM
        if i == 0:
            Wc = norm_weight(dimctx, dim*3)
            params[pp(prefix, 'Wc'+suffix)] = Wc
            Wcx = norm_weight(dimctx, dim)
            params[pp(prefix, 'Wcx'+suffix)] = Wcx
            if options['layer_normalisation']:
                params[pp(prefix,'Wc%s_lnb') % suffix] = scale_add * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wc%s_lns') % suffix] = scale_mul * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_lnb') % suffix] = scale_add * numpy.ones((1*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_lns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix,'Wc%s_wns') % suffix] = scale_mul * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_wns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)          

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att

    if options['layer_normalisation']:
        # layer-normalization parameters
        params[pp(prefix,'W_lnb')] = scale_add * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'W_lns')] = scale_mul * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'U_lnb')] = scale_add * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'U_lns')] = scale_mul * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'Wx_lnb')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Wx_lns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_lnb')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_lns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'W_comb_att_lnb')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'W_comb_att_lns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_lnb')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_lns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    if options['weight_normalisation']:
        params[pp(prefix,'W_wns')] = scale_mul * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'U_wns')] = scale_mul * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'Wx_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'W_comb_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'U_att_wns')] = scale_mul * numpy.ones((1*1)).astype(floatX)

    return params


def lstm_cond_layer(tparams, state_below, options, dropout, prefix='lstm',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   pctx_=None,
                   recurrence_transition_depth=2,
                   truncate_gradient=-1,
                   profile=False,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num= 1 + 2 * recurrence_transition_depth)
    
    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']:
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    below_dropout = dropout((n_samples, dim_below),  dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2*options['dim']), dropout_probability_ctx, num=4)

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim*2))

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    if pctx_ is None:
        pctx_ = tensor.dot(context*ctx_dropout[0], wn(pp(prefix, 'Wc_att'))) +\
            tparams[pp(prefix, 'b_att')]

    if options['layer_normalisation']:
        pctx_ = layer_norm(pctx_, tparams[pp(prefix,'Wc_att_lnb')], tparams[pp(prefix,'Wc_att_lns')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the previous output word embedding
    state_belowx = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'Wx'))) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'W'))) +\
        tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout):
        if options['layer_normalisation']:
            x_ = layer_norm(x_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
            xx_ = layer_norm(xx_, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])

        h_prev = _slice(h_, 0, dim)
        c_prev = _slice(h_, 1, dim)

        preact1 = tensor.dot(h_prev*rec_dropout[0], wn(pp(prefix, 'U')))
        if options['layer_normalisation']:
            preact1 = layer_norm(preact1, tparams[pp(prefix, 'U_lnb')], tparams[pp(prefix, 'U_lns')])
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        i1 = _slice(preact1, 0, dim)
        f1 = _slice(preact1, 1, dim)
        o1 = _slice(preact1, 2, dim)

        preactx1 = tensor.dot(h_prev*rec_dropout[1], wn(pp(prefix, 'Ux')))
        if options['layer_normalisation']:
            preactx1 = layer_norm(preactx1, tparams[pp(prefix, 'Ux_lnb')], tparams[pp(prefix, 'Ux_lns')])
        preactx1 += xx_

        c1 = tensor.tanh(preactx1)
        c1 = f1 * c_prev + i1 * c1
        h1 = o1 * tensor.tanh(c1)

        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_prev
        c1 = m_[:, None] * c1 + (1. - m_)[:, None] * c_prev

        # attention
        pstate_ = tensor.dot(h1*rec_dropout[2], wn(pp(prefix, 'W_comb_att')))
        if options['layer_normalisation']:
            pstate_ = layer_norm(pstate_, tparams[pp(prefix, 'W_comb_att_lnb')], tparams[pp(prefix, 'W_comb_att_lns')])
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__*ctx_dropout[1], wn(pp(prefix, 'U_att')))+tparams[pp(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        h2_prev = h1
        c2_prev = c1
        for i in xrange(recurrence_transition_depth - 1):
            suffix = '' if i == 0 else ('_drt_%s' % i)

            preact2 = tensor.dot(h2_prev*rec_dropout[3+2*i], wn(pp(prefix, 'U_nl'+suffix)))+tparams[pp(prefix, 'b_nl'+suffix)]
            if options['layer_normalisation']:
                preact2 = layer_norm(preact2, tparams[pp(prefix, 'U_nl%s_lnb' % suffix)], tparams[pp(prefix, 'U_nl%s_lns' % suffix)])
            if i == 0:
                ctx1_ = tensor.dot(ctx_*ctx_dropout[2], wn(pp(prefix, 'Wc'+suffix))) # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx1_ = layer_norm(ctx1_, tparams[pp(prefix, 'Wc%s_lnb' % suffix)], tparams[pp(prefix, 'Wc%s_lns' % suffix)])
                preact2 += ctx1_
            preact2 = tensor.nnet.sigmoid(preact2)

            i2 = _slice(preact2, 0, dim)
            f2 = _slice(preact2, 1, dim)
            o2 = _slice(preact2, 2, dim)

            preactx2 = tensor.dot(h2_prev*rec_dropout[4+2*i], wn(pp(prefix, 'Ux_nl'+suffix)))+tparams[pp(prefix, 'bx_nl'+suffix)]
            if options['layer_normalisation']:
               preactx2 = layer_norm(preactx2, tparams[pp(prefix, 'Ux_nl%s_lnb' % suffix)], tparams[pp(prefix, 'Ux_nl%s_lns' % suffix)])
            if i == 0:
               ctx2_ = tensor.dot(ctx_*ctx_dropout[3], wn(pp(prefix, 'Wcx'+suffix))) # dropout mask is shared over mini-steps
               if options['layer_normalisation']:
                   ctx2_ = layer_norm(ctx2_, tparams[pp(prefix, 'Wcx%s_lnb' % suffix)], tparams[pp(prefix, 'Wcx%s_lns' % suffix)])
               preactx2 += ctx2_

            c2 = tensor.tanh(preactx2)
            c2 = f2 * c2_prev + i2 * c2
            h2 = o2 * tensor.tanh(c2)

            h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h2_prev
            h2_prev = h2
            c2_prev = c2

        h2 = concatenate([h2, c2], axis=1)

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = []

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.zeros((n_samples,
                                                               context.shape[2])),
                                                  tensor.zeros((n_samples,
                                                               context.shape[0]))],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_gradient,
                                    profile=profile,
                                    strict=False)
    return rval
