import re
import pickle

def normalize(text):
    def insertSpace(token, text):
        sidx = 0
        while True:
            sidx = text.find(token, sidx)
            if sidx == -1:
                break
            if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                    re.match('[0-9]', text[sidx + 1]):
                sidx += 1
                continue
            if text[sidx - 1] != ' ':
                text = text[:sidx] + ' ' + text[sidx:]
                sidx += 1
            if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
                text = text[:sidx + 1] + ' ' + text[sidx + 1:]
            sidx += 1
        return text
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)


    text = ' ' + text + ' '
    # # replace time and and price
    timepat = re.compile(" \d{1,2}[:]\d{1,2}[ \.,\?]| \d{4}[ \.,\?]| \d{1,2}[ap][m\. ]+| \d{1,2} [ap][m\. ]+| \d{1,2}[:]\d{1,2}[ap]m[ \.,\?]")
    # # some utterances just miss the ":"
    # timepat_noise = re.compile(" at \d{4}[ \.,\?]")
    pricepat = re.compile("\d{1,3}[.]\d{1,2}")

    value_time = re.findall(timepat, text)

    while value_time:
        index = text.find(value_time[0])
        text = text[:index] + \
                ' [value_time|' + value_time[0][1:-1] + ']' + \
                text[index+len(value_time[0]) - 1:]
        value_time = re.findall(timepat, text)

    value_price = re.findall(pricepat, text)

    if value_price:
        text = re.sub(pricepat, ' [value_price|' + value_price[0] + '] ', text)

    text = text[1:-1]

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)

    fin = open('./simulator/multiwoz_utils/utils/mapping.pair')
    replacements = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def check_balance(string):
    # open_tup = tuple('[') 
    # close_tup = tuple(']') 
    # map = dict(zip(open_tup, close_tup)) 
    queue = 0
    
    for i in string: 
        if i == '[': 
            queue += 1
        elif i == ']': 
            if not queue: 
                return False
            else:
                queue -= 1
    if not queue:
        return True
    else:
        return False

def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ')
        if key in utt:
            idx = 0
            while utt[idx:].find(' ' + key + ' ') != -1:
                idx += utt[idx:].find(' ' + key + ' ')
                # # to exclude the case that 'ask' is a verb
                if key == 'ask' and idx > 2 and utt[idx-2:idx] == ' i':
                    idx += 1
                    continue
                if check_balance(utt[:idx]):
                    utt = utt[:idx] + ' ' + val[:-1] + '|' + key + '] ' + utt[idx + len(key) + 2:]
                    idx += len(key) + 4 + len(val[:-1])
                else:
                    idx += len(key)
        utt = utt[1:-1]

    return utt

def delex_people_count(sent):
    sent = ' ' + sent + ' '
    digitpat = re.compile('(?<!looking for)(?<=for) \d+ (?!of)|(?<=party of) \d+ | \d+ (?=people|person|of us)')
    value_people = re.findall(digitpat, sent)
    while value_people:
        index = sent.find(value_people[0])
        if not check_balance(sent[:index]):
            value_people.pop(0)
            continue

        sent = sent[:index] + \
                ' [value_people|' + value_people[0][1:-1] + '] ' + \
                sent[index+len(value_people[0]):]
        value_people = re.findall(digitpat, sent)
    sent = sent[1:-1]
    return sent

def delex_sent(response):
    # # replace time, date, specific price
    response = normalize(response)

    # # replace info in db
    db_entity_file = open('./data/multiwoz-master/db_entity_file.pkl','rb')
    db_entity_list = pickle.load(db_entity_file)
    db_entity_file.close()
    response = delexicalise(response, db_entity_list)

    # # replace # of people for reservation
    response = delex_people_count(response)
    return response
