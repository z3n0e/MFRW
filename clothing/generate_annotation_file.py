import pickle

clean_kv = 'annotations/clean_label_kv.txt'

with open(clean_kv) as f:
    lines_clean_kv = [x.strip() for x in f.readlines()]

dict_clean_kv = {}

for el in lines_clean_kv:
    el_split = el.split()
    dict_clean_kv[el_split[0]] = el_split[1]

### generate dict for test example
test_list = 'annotations/clean_test_key_list.txt'
with open(test_list) as f:
    lines_test_list = [x.strip() for x in f.readlines()]

dict_test = {}

for el in lines_test_list:
    dict_test[el] = dict_clean_kv[el]

test_kv = 'annotations/test_kv.txt'

with open('annotations/test_kv.pickle', 'wb') as handle:
    pickle.dump(dict_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

test_file = open(test_kv, "w")

for element in dict_test:
    print(element)
    test_file.write(element + ' ' + str(dict_test[element]) + '\n')

test_file.close()

### generate dict for validation example
val_list = 'annotations/clean_val_key_list.txt'
with open(val_list) as f:
    lines_val_list = [x.strip() for x in f.readlines()]

dict_val = {}

for el in lines_val_list:
    dict_val[el] = dict_clean_kv[el]

val_kv = 'annotations/val_kv.txt'

with open('annotations/val_kv.pickle', 'wb') as handle:
    pickle.dump(dict_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

val_file = open(val_kv, "w")

for element in dict_val:
    print(element)
    val_file.write(element + ' ' + str(dict_val[element]) + '\n')

val_file.close()

### generate dict for train example
train_list = 'annotations/clean_train_key_list.txt'
with open(train_list) as f:
    lines_train_list = [x.strip() for x in f.readlines()]

dict_train = {}

for el in lines_train_list:
    dict_train[el] = dict_clean_kv[el]

print(dict_train)

noisy_kv = 'annotations/noisy_label_kv.txt'

with open(noisy_kv) as f:
    lines_noisy_kv = [x.strip() for x in f.readlines()]

dict_noisy_kv = {}

for el in lines_noisy_kv:
    el_split = el.split()
    dict_noisy_kv[el_split[0]] = el_split[1]

print(dict_noisy_kv)

train_noisy_list = 'annotations/noisy_train_key_list.txt'

with open(train_noisy_list) as f:
    lines_noisy_train = [x.strip() for x in f.readlines()]

for el in lines_noisy_train:
    if el in dict_train:
        print('ce già')
    if el not in dict_train:
        print('non ce già')
        dict_train[el] = dict_noisy_kv[el]

train_kv = 'annotations/train_kv.txt'

with open('annotations/train_kv.pickle', 'wb') as handle:
    pickle.dump(dict_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

train_file = open(train_kv, "w")

for element in dict_train:
    print(element)
    train_file.write(element + ' ' + str(dict_train[element]) + '\n')

train_file.close()
