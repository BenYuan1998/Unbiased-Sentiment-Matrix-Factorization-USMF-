
def get_dict(path):
    dict = {}
    with open(path) as data_file:
        for line in data_file:
            dict[line.strip()] = ""
    return dict

def classification(feature, word, neg_dict, pos_dict, count):
    entry = feature + " : " + word
    result = raw_input(str(count) + " " + entry + "\nPlease classify: ")
    if result == "a":
        pos_dict[entry] = ""
    elif result == "d":
        neg_dict[entry] = ""
    elif result == "s":
        return
    else:
        classification(feature, word, neg_dict, pos_dict, count)

def auto_classify(feature, word, neg_words, pos_words, neg_entries, pos_entries):
    # add not rules
    neg_set = ["not", "n't"]
    if len(word.split()) > 1 and word.split()[0] in neg_set:
        word = word.split()[1]
    entry = feature + ":" + word
    if word in pos_words:
        pos_entries[entry] = ""
    elif word in neg_words:
        neg_entries[entry] = ""


neg_copus = "../data/Cell_Phones_and_Accessories_5/opinion-lexicon-English/negative-words.txt"
pos_copus = "../data/Cell_Phones_and_Accessories_5/opinion-lexicon-English/positive-words.txt"
candidate_path = "../data/Cell_Phones_and_Accessories_5/all_lexicon.txt"
feature_path = "../data/Cell_Phones_and_Accessories_5/feature-cluster.txt"
feature_set = {}
neg_entries = {}
pos_entries = {}
neg_words = get_dict(neg_copus)
pos_words = get_dict(pos_copus)
with open(feature_path) as data_file:
    for line in data_file:
        aspect = line.split(":")[0]
        features = line.split(":")[1]
        feature_list = features.strip().split(",")
        for feature in feature_list:
            feature = feature.strip()
            feature_set[feature] = ""
count = 1
with open(candidate_path) as data_file:
    for line in data_file:
        feature = line.split(":")[0].strip()
        if feature not in feature_set:
            continue
        words = line.split(":")[1].split(",")
        for word in words:
            word = word.strip()
            if word == "":
                continue
            count += 1
print("Totally " + str(count) + " words.")
with open(candidate_path) as data_file:
    for line in data_file:
        feature = line.split(":")[0].strip()
        if feature not in feature_set:
            continue
        words = line.split(":")[1].split(",")
        for word in words:
            word = word.strip()
            if word == "":
                continue
            auto_classify(feature, word, neg_words, pos_words, neg_entries, pos_entries)
with open('neg.txt', 'w') as file:
    for key in neg_entries.keys():
        file.write(key + "\n")
with open('pos.txt', 'w') as file:
    for key in pos_entries.keys():
        file.write(key + "\n")

