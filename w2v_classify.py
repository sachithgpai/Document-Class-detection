import gensim
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from os import listdir

# System parameters.
num_features = 30			# Dimension of the vector space to which the documents will be appended.
num_workers =4				# number of threads.(Word2vec parameter)

#Right now the we have one file per class. Minor modifications can be done to read from multiple files

filename_class1 = "LitreatureDataset.txt"	# The first document class.
filename_class2 = "NewsDataset.txt"			# The second Document class.
test_files = [f for f in listdir(".\\Test\\")] # The test files are placed in a folder 'Test' in the same path


#takes each sentence and removes unwanted things.
def line_to_words(line):
	line.replace("<br \>","")
	line = re.sub("[^a-zA-Z]", " ", line)
	words = line.lower().split()
	stops = set(stopwords.words("english"	))
	words = [w for w in words if not w in stops]
	return (words)


#This function trains a Word2Vec model and returns the object.
def train_model(filename,save_model):
	f =[f for f in listdir(".\\")]
	if ('w2v_'+filename) in f:
		model = gensim.models.Word2Vec.load('w2v_'+filename)
		return model

	text_file = open(filename, "r")
	lines = text_file.read().split(".")
	sentences = []
	for raw_sent in lines:
		sent = line_to_words(raw_sent)
		sentences.append(sent)
	model = gensim.models.Word2Vec(sentences, min_count=1, workers=num_workers, size=num_features)

	if(save_model):
		model.save('w2v_'+filename)
	return model



#This is a similarity measure to compare the closeness between the embeddings.
#What it is doing is taking the intersection of vocabularies of the two embeddings and
# for each word in this vocab_intersection check how many of the k-nearest neighbour words match for each word
# between the two embedding. The score added is:
# (number of words in the KNN of both embeddings)/(number of unique words appearing in either of KNNs)
def similarity_of_models(model1,model2):
	vocab1 = model1.wv.vocab.keys()
	vocab2 = model2.wv.vocab.keys()
	KNN_cnt = 50

	vocab_intersection = [val for val in vocab1 if val in vocab2]

	sum =0.0
	for word in vocab_intersection:
		result1 = [item[0] for item in model1.similar_by_word(word, topn=KNN_cnt) if item[0] in vocab_intersection]
		result2 = [item[0] for item in model2.similar_by_word(word, topn=KNN_cnt) if item[0] in vocab_intersection]

		numer = len([item for item in result1 if item in result2])
		denom = len( result1 + [e for e in result2 if e not in result1])
		if denom>0:
			sum = sum + (numer*1.0/denom)

	return sum/len(vocab_intersection)




def plot_test_points(mod_A_sim,mod_B_sim):
	num_news = len(test_files)/2					#We have equal number of test files from each class.
	plt.scatter(mod_A_sim[:num_news], mod_B_sim[:num_news], color='green', label="News")
	plt.scatter(mod_A_sim[num_news:], mod_B_sim[num_news:], color='red', label="Literature")
	plt.plot([0, 1], [0, 1], alpha=0.3)
	plt.xlim(0, 0.02)
	plt.ylim(0, 0.02)
	plt.xlabel("Similarity to Literature")
	plt.ylabel("Similarity to News")
	plt.legend()
	plt.show()




def main():
	model_A = train_model(filename_class1,1)
	model_B = train_model(filename_class2,1)

	A_vocab_sz= len(model_A.wv.vocab.keys())
	B_vocab_sz = len(model_B.wv.vocab.keys())
	ratio = A_vocab_sz*1.0/B_vocab_sz
	print 'Model Literature Vocab size:',A_vocab_sz
	print 'Model News Vocab size:',B_vocab_sz

	mod_A_sim = []
	mod_B_sim = []
	cnt_missclass_news=0
	cnt_missclass_lit=0
	for file in test_files:
		testmodel = train_model(".\\Test\\"+file,0)
		sim_to_modelA = similarity_of_models(model_A, testmodel)
		sim_to_modelB = similarity_of_models(model_B, testmodel)
		if(sim_to_modelA>sim_to_modelB):
			cnt_missclass_lit+=(test_files.index(file)<30)
		else:
			cnt_missclass_news += (test_files.index(file) >= 30)
		mod_A_sim.append(sim_to_modelA)
		mod_B_sim.append(sim_to_modelB*ratio)			#Scaling the similarity to account for different vocabulary size.

	print "Accuracy: ",(len(test_files)-cnt_missclass_news-cnt_missclass_lit)/len(test_files)
	print "Percent Missclassified as News", cnt_missclass_news/len(test_files)
	print "Percent Missclassified as Literature", cnt_missclass_lit	 / len(test_files)

	plot_test_points(mod_A_sim,mod_B_sim)



if __name__ == "__main__":
        main()