"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        PointerRNNDecoder, HierarchicalEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder
from onmt.Utils import use_gpu


def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True, num_interval=5, char_dict=None):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    feat_vec_size = opt.feat_vec_size
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
        
        if opt.hier_model1 and opt.hier_two_dim_record and opt.two_dim_concat:
            assert embedding_dim % 2 == 0 and feat_vec_size % 2 == 0
            embedding_dim = embedding_dim // 2
            feat_vec_size = feat_vec_size // 2
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    char_padding_idx = char_dict.stoi[onmt.io.PAD_WORD] if char_dict is not None else None
    num_word_embeddings = len(word_dict)
    num_char_embeddings = len(char_dict) if char_dict is not None else None

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    if char_dict is not None:
        bi_flag = True if opt.char_bidirectional else False
    else:
        bi_flag = None

    if for_encoder:
        print("Encoder: status for mix rank info is interval {}, num of char embedding {} and biRNN {}.".format(num_interval, num_char_embeddings, bi_flag))
    else:
        print("Decoder: status for mix rank info is interval {}, num of char embedding {} and biRNN {}.".format(num_interval, num_char_embeddings, bi_flag))

    main_emb = Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      num_interval=num_interval,
                      char_vocab_size=num_char_embeddings, 
                      char_padding_idx=char_padding_idx, 
                      char_vec_size=opt.char_vec_size if char_dict is not None else None, 
                      char_rnn_type=opt.char_rnn_type if char_dict is not None else None, 
                      char_hidden_size=opt.char_rnn_size if char_dict is not None else None, 
                      char_num_layers=opt.char_layers if char_dict is not None else None, 
                      char_dropout=opt.char_dropout if char_dict is not None else None, 
                      char_bidirection=bi_flag)

    return main_emb

def make_encoder(opt, embeddings, stage1=True, basic_enc_dec=False, no_cs_gate=False):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
        stage1: stage1 encoder
    """

    if stage1 or basic_enc_dec:
        if opt.hier_model1:
            try:
                cohan18=opt.cohan18
            except:
                cohan18 = None

            try:
                cohan18_hidden_size = opt.cohan18_hidden_size
            except:
                cohan18_hidden_size = None
            return HierarchicalEncoder(opt.hier_meta, opt.enc_layers1, embeddings, opt.src_word_vec_size, opt.attn_hidden, 
                                       opt.hier_rnn_type, opt.hier_bidirectional, opt.hier_rnn_size, dropout=opt.dropout, 
                                       attn_type=opt.global_attention, two_dim_record=opt.hier_two_dim_record if opt.hier_two_dim_record is not None else False, 
                                       hier_row_rnn=opt.hier_row_rnn if opt.hier_row_rnn is not None else False, hier_col_dim_rnn=opt.hier_col_dim_rnn if opt.hier_col_dim_rnn is not None else False, 
                                       hier_record_level_use_attn=opt.hier_record_level_use_attn if opt.hier_record_level_use_attn is not None else False, 
                                       two_dim_use_mlp=opt.two_dim_use_mlp if opt.two_dim_use_mlp is not None else False, two_dim_concat=opt.two_dim_concat if opt.two_dim_concat is not None else False, 
                                       two_dim_gate_direct_scalar=opt.two_dim_gate_direct_scalar if opt.two_dim_gate_direct_scalar is not None else False, 
                                       two_dim_gate_activation_scalar=opt.two_dim_gate_activation_scalar if opt.two_dim_gate_activation_scalar is not None else False, 
                                       row_self_attn_type=opt.row_self_attn_type, col_self_attn_type=opt.col_self_attn_type, multi_head_count=opt.multi_head_count, 
                                       multi_head_dp=opt.multi_head_dp, true_two_dim_fusion=opt.true_two_dim_fusion, no_gate=opt.gsa_no_gate if opt.gsa_no_gate is not None else False, 
                                       residual=opt.gsa_residual if opt.gsa_residual is not None else False, mha_concat=opt.mha_concat if opt.mha_concat is not None else False, 
                                       mha_residual=opt.mha_residual if opt.mha_residual is not None else False, mha_norm=opt.mha_norm if opt.mha_norm is not None else False, 
                                       two_dim_softmax=opt.two_dim_softmax if opt.two_dim_softmax is not None else False, two_dim_gen_tanh=opt.two_dim_gen_tanh if opt.two_dim_gen_tanh is not None else False, 
                                       two_dim_score=opt.two_dim_score, no_gate_relu=opt.gsa_no_gate_relu if opt.gsa_no_gate_relu is not None else False, 
                                       no_gate_bias=opt.gsa_no_gate_bias if opt.gsa_no_gate_bias is not None else False, only_ply_level_gate=opt.only_ply_level_gate if opt.only_ply_level_gate is not None else False, 
                                       true_hier_record_level_use_attn=opt.true_hier_record_level_use_attn if opt.true_hier_record_level_use_attn is not None else False, 
                                       hier_bi=opt.hier_bi if opt.hier_bi is not None else False, hier_num_layers=opt.hier_num_layers, 
                                       norow=opt.norow if opt.norow is not None else False, nocolumn=opt.nocolumn if opt.nocolumn is not None else False, 
                                       nofusion=opt.nofusion if opt.nofusion is not None else False, nohierstructure=opt.nohierstructure if opt.nohierstructure is not None else False, 
                                       cohan18=cohan18 if cohan18 is not None else False, nohistpos=opt.nohistpos if opt.nohistpos is not None else False, cohan18_hidden_size=opt.cohan18_hidden_size, 
                                       is_cnn_cell=opt.cnncell if opt.cnncell is not None else False, cnn_kernel_width=opt.cnn_kernel_width, 
                                       mlp_hier=opt.mlp_hier if opt.mlp_hier is not None else False, two_mlp_hier=opt.two_mlp_hier if opt.two_mlp_hier is not None else False)
        else:
            return MeanEncoder(opt.hier_meta, opt.enc_layers1, embeddings, opt.src_word_vec_size, opt.attn_hidden, opt.dropout, enable_attn=not basic_enc_dec and not no_cs_gate, no_gate=opt.gsa_no_gate if opt.gsa_no_gate is not None else False, residual=opt.gsa_residual if opt.gsa_residual is not None else False, no_gate_relu=opt.gsa_no_gate_relu if opt.gsa_no_gate_relu is not None else False, no_gate_bias=opt.gsa_no_gate_bias if opt.gsa_no_gate_bias is not None else False, mode=opt.mean_enc_mode if opt.mean_enc_mode is not None else "MLP", trans_layer=opt.mean_trans_layer, rnn_type=opt.mean_rnn_type, bidirectional=opt.mean_rnn_bi if opt.mean_rnn_bi is not None else False, hidden_size=opt.mean_hidden, cnn_kernel_width=opt.cnn_kernel_width, multi_head_count=opt.multi_head_count, multi_head_dp=opt.multi_head_dp)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn2, opt.enc_layers2,
                          opt.rnn_size, opt.dropout, embeddings,
                          opt.bridge)


def make_decoder(opt, embeddings, stage1, basic_enc_dec):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
        stage1: stage1 decoder
    """
    if stage1:
        return PointerRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers1, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             False,
                             opt.dropout,
                             embeddings,
                             False,
                             opt.decoder_type1, 
                             opt.hier_model1 and not opt.nohierstructure)
    else:
        try:
            cohan18=opt.cohan18
        except:
            cohan18 = None
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn2,
                                   opt.dec_layers2, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   True,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn,
                                   hier_attn=opt.hier_model1 and not opt.nohierstructure if opt.hier_model1 is not None and basic_enc_dec else False,
                                   hier_mix_attn=opt.hier_mix_attn if opt.hier_mix_attn is not None else False, cohan18=cohan18)

def load_test_model(opt, dummy_opt, stage1=False):
    opt_model = opt.model if stage1 else opt.model2
    checkpoint = torch.load(opt_model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint, stage1, model_opt.basicencdec)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None, stage1=True, basic_enc_dec=False):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    if stage1 and not basic_enc_dec:
        src = "src1"
        src_char = "src1_char"
        tgt = "tgt1"
    else:
        src = "src2"
        src_char = "src2_char"
        tgt = "tgt2"
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields[src].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, src)
        src_char_dict = fields[src_char].vocab if model_opt.char_enable else None

        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts, num_interval=model_opt.num_interval, char_dict=src_char_dict)
        tgt_dict = fields[tgt].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, tgt)
        ref_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

        no_cs_gate_flag = model_opt.nocsgate if model_opt.nocsgate is not None else False
        encoder = make_encoder(model_opt, src_embeddings, stage1, basic_enc_dec, no_cs_gate_flag)
        ref_encoder = make_encoder(model_opt, ref_embeddings, False, False, no_cs_gate_flag)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields[tgt].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, tgt)
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings, stage1 and not basic_enc_dec, basic_enc_dec)

    # Make NMTModel(= encoder + ref_encoder + decoder).
    model = NMTModel(encoder, ref_encoder, decoder)
    model.model_type = model_opt.model_type

    # Make Generator.
    if stage1 and not basic_enc_dec:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt1"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt2"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
