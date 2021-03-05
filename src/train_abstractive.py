#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch
from pytorch_transformers import BertTokenizer

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.loss import abs_loss
from models.model_builder import AbsSummarizer
from models.predictor import build_predictor
from models.trainer import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

def test_text_abs(args):

    logger.info('Loading checkpoint from %s' % args.test_from)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.load_text(args, args.text_src, args.text_tgt, device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, -1)
