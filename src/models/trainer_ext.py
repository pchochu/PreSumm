import numpy as np
import torch

def build_trainer(args, device_id, model):
    trainer = Trainer(args, model)
    return trainer
class Trainer(object):
    def __init__(self, args, model):
        # Basic attributes.
        self.args = args
        self.model = model
        self.model.eval()

    def test(self, test_iter, step):

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        with torch.no_grad():
            pred = []
            for batch in test_iter:
                src = batch.src
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()
                selected_ids = np.argsort(-sent_scores, 1)

                for i, idx in enumerate(selected_ids):
                    _pred = []
                    if (len(batch.src_str[i]) == 0):
                        continue
                    for j in selected_ids[i][:len(batch.src_str[i])]:
                        if (j >= len(batch.src_str[i])):
                            continue
                        candidate = batch.src_str[i][j].strip()
                        if (self.args.block_trigram):
                            if (not _block_tri(candidate, _pred)):
                                _pred.append(candidate)
                        else:
                            _pred.append(candidate)

                        if ((not self.args.recall_eval) and len(_pred) == 5):
                            break

                    _pred = ' '.join(_pred)
                    if (self.args.recall_eval):
                        _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                    pred.append(_pred)
            return pred