from string import digits
import re


emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub('http://\S+|https://\S+', '', text)
    text = text.replace(u',', '')
    text = text.replace(u'"', '')
    text = text.replace(u'(', '')
    text = text.replace(u')', '')
    text = text.replace(u'"', '')
    text = text.replace(u':', '')
    text = text.replace(u"'", '')
    text = text.replace(u"‘‘", '')
    text = text.replace(u"’’", '')
    text = text.replace(u"''", '')
    text = text.replace(u'-', '')
    text = text.replace(u"\n","")
    text = text.replace(u'.', '')
    text = text.replace(u"?","")
    text = text.replace(u"\\n","")
    text = text.replace(u"[", "")
    text = text.replace(u"]", "")
    text = text.replace(u'#', "")
    text = text.replace(u'@', "")
    text = re.sub(r'[0-9]+', '', text)
    text = emoji_pattern.sub(r'', text)

    return text


def get_class_dicts():

    a = ['activate', 'bring', 'change language', 'deactivate', 'decrease',
        'increase']
    o = ['Chinese', 'English', 'German', 'Korean', 'heat', 'juice', 'lamp',
            'lights', 'music', 'newspaper', 'none', 'shoes', 'socks', 'volume']

    l = ['bedroom', 'kitchen', 'none', 'washroom']

    class_dict_a = {}
    class_dict_o = {}
    class_dict_l = {}

    for num, i in enumerate(a):
        class_dict_a[i] = num

    for num, j in enumerate(o):
        class_dict_o[j] = num

    for num, k in enumerate(l):
        class_dict_l[k] = num

    return class_dict_a, class_dict_o, class_dict_l

