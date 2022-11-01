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
        return self.mfc


class NB:
    def __init__(self, klasses, docs):
        self.pos = {}
        self.neg = {}
        self.neu = {}
        self.npos = 0
        self.nneg = 0
        self.nneu = 0
        self.posDocs = 0
        self.negDocs = 0
        self.neuDocs = 0
        self.i = 0
        self.p = {}
        self.p['pos'] = {}
        self.p['neg'] = {}
        self.p['neu'] = {}
        self.train(klasses, docs)

    def train(self, klasses, docs):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k, d in zip(klasses, docs):
            tokens = tokenize(d)
            self.i += 1
            if k == 'positive':
                self.posDocs += 1
            if k == 'negative':
                self.negDocs += 1
            if k == 'neutral':
                self.neuDocs += 1

            for t in tokens:
                if k == 'positive':
                    self.npos += 1
                    if t in self.pos:
                        self.pos[t] += 1
                    else:
                        self.pos[t] = 1
                if k == 'negative':
                    self.nneg += 1
                    if t in self.neg:
                        self.neg[t] += 1
                    else:
                        self.neg[t] = 1
                if k == 'neutral':
                    self.nneu += 1
                    if t in self.neu:
                        self.neu[t] += 1
                    else:
                        self.neu[t] = 1
        for pos in self.pos:
            self.p['pos'][pos] = (self.pos[pos]+1)/(self.npos + len(self.pos))
        for neg in self.neg:
            self.p['neg'][neg] = (self.neg[neg]+1)/(self.nneg + len(self.neg))
        for neu in self.neu:
            self.p['neu'][neu] = (self.neu[neu]+1) / (self.nneu + len(self.pos))

    def classify(self, tweet):
        tokens = tokenize(tweet)
        probPos = 0
        probNeg = 0
        probNeu = 0
        for t in tokens:
            if t in self.p['pos']:
                probPos += math.log(self.p['pos'][t])
            if t in self.p['neg']:
                probNeg += math.log(self.p['neg'][t])
            if t in self.p['neu']:
                probNeu += math.log(self.p['neu'][t])
        probPos = math.log(self.posDocs/self.i)+probPos
        probNeg = math.log(self.negDocs / self.i) + probNeg
        probNeu = math.log(self.neuDocs / self.i) + probNeu

        if probPos > probNeg and probPos > probNeu:
            return 'positive'
        if probNeg > probPos and probNeg > probNeu:
            return 'negative'
        if probNeu > probPos and probNeu > probNeg:
            return 'neutral'
class NBBIN:
    def __init__(self, klasses, docs):
        self.pos = {}
        self.neg = {}
        self.neu = {}
        self.npos = 0
        self.nneg = 0
        self.nneu = 0
        self.posDocs = 0
        self.negDocs = 0
        self.neuDocs = 0
        self.i = 0
        self.p = {}
        self.p['pos'] = {}
        self.p['neg'] = {}
        self.p['neu'] = {}
        self.train(klasses, docs)

    def train(self, klasses, docs):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k, d in zip(klasses, docs):
            tokens = tokenize(d)
            self.i += 1
            if k == 'positive':
                self.posDocs += 1
            if k == 'negative':
                self.negDocs += 1
            if k == 'neutral':
                self.neuDocs += 1

            for t in tokens:
                if k == 'positive':
                    self.npos += 1
                    if t not in self.pos:
                        self.pos[t] = 1
                if k == 'negative':
                    self.nneg += 1
                    if t not in self.neg:
                        self.neg[t] = 1
                if k == 'neutral':
                    self.nneu += 1
                    if t not in self.neu:
                        self.neu[t] = 1
        for pos in self.pos:
            self.p['pos'][pos] = (self.pos[pos]+1)/(self.npos + len(self.pos))
        for neg in self.neg:
            self.p['neg'][neg] = (self.neg[neg]+1)/(self.nneg + len(self.neg))
        for neu in self.neu:
            self.p['neu'][neu] = (self.neu[neu]+1) / (self.nneu + len(self.pos))

    def classify(self, tweet):
        tokens = tokenize(tweet)
        probPos = 0
        probNeg = 0
        probNeu = 0
        for t in tokens:
            if t in self.p['pos']:
                probPos += math.log(self.p['pos'][t])
            if t in self.p['neg']:
                probNeg += math.log(self.p['neg'][t])
            if t in self.p['neu']:
                probNeu += math.log(self.p['neu'][t])
        probPos = math.log(self.posDocs/self.i)+probPos
        probNeg = math.log(self.negDocs / self.i) + probNeg
        probNeu = math.log(self.neuDocs / self.i) + probNeu

        if probPos > probNeg and probPos > probNeu:
            return 'positive'
        if probNeg > probPos and probNeg > probNeu:
            return 'negative'
        if probNeu > probPos and probNeu > probNeg:
            return 'neutral'

class Lexicon:

    def classify(self, tweet):
        with open('pos-words.txt') as pfile:
            with open("neg-words.txt") as nfile:
                tokens = None
                positives = pfile.read()
                negitives = nfile.read()
                i = 0
                #out = []
                #for tweet in test_instance:
                tokens = tokenize(tweet)
                p = 0
                n = 0
                for tok in tokens:
                    if(tok in positives):
                        p += 1
                    if(tok in negitives):
                        n += 1
                if p > n:
                    return "positive"
                elif p < n:
                    return "negative"
                else:
                    return "neutral"


if __name__ == '__main__':
    import sys

    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'lexicon', 'nb', or
    # 'nbbin'
    # method = sys.argv[1]
    #
    train_texts_fname = 'train.docs.txt'#sys.argv[2]
    train_klasses_fname = 'train.classes.txt'#sys.argv[3]
    test_texts_fname = 'dev.docs.txt'#sys.argv[4]
    #
    train_texts = [x.strip() for x in open(train_texts_fname,
                                            encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                              encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                           encoding='utf8')]
    method = 'nbbin'
    if method == 'lexicon':
        classifier = Lexicon()
        results = [classifier.classify(x) for x in test_texts]

    if method == 'nb':
        classifier = NB(train_klasses, train_texts)
        results = [classifier.classify(x) for x in test_texts]
    if method == 'nbbin':
        classifier = NBBIN(train_klasses, train_texts)
        results = [classifier.classify(x) for x in test_texts]

    for r in results:
        print(r)

