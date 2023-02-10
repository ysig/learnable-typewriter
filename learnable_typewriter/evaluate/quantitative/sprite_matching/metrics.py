import itertools
import editdistance

def list_split(x, delim):
    return [tuple(y) for a, y in itertools.groupby(x, lambda z: z == delim) if not a]

def format_text(txt, sep):
    try:
        txt = sep.join(txt)
    except TypeError:
        txt = sep.join(map(str, txt))
    return txt

def cer(pd, gt, delim):
    pd = [p for p in pd if p != delim]
    gt = [g for g in gt if g != delim]
    dist = editdistance.eval(pd, gt)
    return dist / len(gt)

def ser(pd, gt):
    return editdistance.eval([tuple(pd)], [tuple(gt)])

def error_rate(data, delim, sep, map_gt=None, map_pd=None, verbose=False, average=False):
    if average:
        cer_acc, ser_acc = 0, 0
    else:
        cer_acc, ser_acc, texts, gts = [], [], [], []

    i = 0
    for (pd, gt) in data:
        if map_gt is not None:
            gt = [map_gt[g] for g in gt]
        if map_pd is not None:
            pd = [map_pd.get(p, '_') for p in pd if p != -1]  

        cer_, ser_ = cer(pd, gt, delim), ser(pd, gt)
        if average:
            cer_acc += cer_
            ser_acc += ser_
        else:
            cer_acc.append(cer_)
            ser_acc.append(ser_)
            texts.append(format_text(pd, sep))
            gts.append(format_text(gt, sep))

        i += 1

    if average:
        output = {'cer': cer_acc/i, 'ser': ser_acc/i}
    else:
        assert len(cer_acc) == len(ser_acc) == len(texts) == len(gts)
        output = {'cer': cer_acc, 'ser': ser_acc, 'texts': texts, 'gt': gts}
    return output


