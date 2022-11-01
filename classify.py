import math, re


# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t)  # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t)  # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens


# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True,
                          key=lambda x: klass_freqs[x])[0]

    def classify(self, test_instance):
        with open('pos-words.txt') as pfile:
            with open("neg-words.txt") as nfile:
                tokens = None
                positives = pfile.read()
                negitives = nfile.read()
                i = 0
                out = []
                for tweet in test_instance:
                    tokens = tokenize(tweet)
                    p = 0
                    n = 0
                    for tok in tokens:
                        if(tok in positives):
                            p += 1
                        elif(tok in negitives):
                            n += 1
                    if(p>n):
                        out[i] = "positive"
                    elif(p<n):
                        out[i] = "negative"
                    else:
                        out[i] = "neutral"


if __name__ == '__main__':
    import sys

    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'lexicon', 'nb', or
    # 'nbbin'
    method = sys.argv[1]

    train_texts_fname = sys.argv[2]
    train_klasses_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]

    train_texts = [x.strip() for x in open(train_texts_fname,
                                           encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                             encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]

    if method == 'baseline':
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_texts]