from lib.utils import TextDataset


label_name = ['World', 'Sports', 'Business', 'Sci/Tech']
training_num, valid_num, test_num, vocab_size = 110000, 10000, 7600, 10000
training_file = 'dataset/agnews_training_110K_10K-TFIDF-words.txt'
valid_file = 'dataset/agnews_valid_10K_10K-TFIDF-words.txt'
test_file = 'dataset/agnews_test_7600_10K-TFIDF-words.txt'

traindata = TextDataset(training_file, training_num, vocab_size)
# testdata = TextDataset(test_file, test_num, vocab_size)
print(traindata[100000])