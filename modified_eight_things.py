import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

#reproducibility
numpy.random.seed(1)
tf.set_random_seed(1)

#set properties of neural net such as size, input, output
nDomains = 4
nItemsPerDomain = 8 #should be 2k for some k >= 2, and 2(k+1) should not be larger than nSingleOutputSize
nContextPerDomain = 4
nSingleOutputSize = 14

nSizeofItemInput = nDomains*nItemsPerDomain #size of item input vector
nSizeofContextInput = nDomains*nContextPerDomain#size of context input vector

nSizeofInput = (nItemsPerDomain+nContextPerDomain)*nDomains #length of a single input vector
nSizeofOutput = nSingleOutputSize*nDomains*nContextPerDomain # length of a single output vector
nNumberOfPossibleInputPatterns = nItemsPerDomain*nContextPerDomain*nDomains #number of training examples in dataset

sizeItemRepLayer = 8
sizeContextRepLayer = 4
sizeSecondLayer = 32

#training properties
nEpochs = 500
eta = 0.001

#construct the input matrix
InputMatrix = numpy.empty([nNumberOfPossibleInputPatterns,nSizeofInput],int)
InputCount = 0

A = numpy.zeros((nItemsPerDomain,nDomains),dtype =numpy.int)
B = numpy.zeros((nContextPerDomain,nDomains),dtype =numpy.int)


for i in range(nDomains):
	for j in range(nItemsPerDomain):
		A[j,i] = 1
		tempA = A.flatten('F')
		for k in range(nContextPerDomain):
			B[k,i] = 1
			tempB = B.flatten('F')
			temp = numpy.concatenate((tempA,tempB))
			InputMatrix[InputCount] = temp
			InputCount = InputCount + 1

			B = numpy.zeros((nContextPerDomain,nDomains))
		A = numpy.zeros((nItemsPerDomain,nDomains))
	

print InputMatrix.shape #each row is an input vector, number of row is number of training example

#plt.imshow(numpy.dot(InputMatrix,InputMatrix.transpose()),cmap='Greys',interpolation='none')
#plt.show()


#construct the output templates
OutputMatrix = numpy.zeros([nNumberOfPossibleInputPatterns,nSizeofOutput],int)
OutputMatrix_ContextCollapsed = numpy.empty([nItemsPerDomain*nDomains,nSizeofOutput],int) #Collapsing across context for similarity computation
OutputCount = 0

output_template = numpy.empty([nItemsPerDomain*nContextPerDomain,nSingleOutputSize]) #template for what the outputs in each domain and context will look like
output_similarity_step_size = nSingleOutputSize//(nItemsPerDomain) #How many entries will be made changed in each pair of items within a context
for ctx_i in xrange(nContextPerDomain):
    basic_output = numpy.random.permutation(numpy.concatenate((numpy.zeros(nSingleOutputSize//2),numpy.ones(nSingleOutputSize-(nSingleOutputSize//2))))) #Random pattern that will form basis for the similarity
    locations = numpy.argsort(basic_output)
    zero_locations = numpy.random.permutation(locations[:nSingleOutputSize//2]) #at each step of similarity will flip one of these and one of the ones
    one_locations = numpy.random.permutation(locations[nSingleOutputSize//2:])
    for out_i in xrange(nItemsPerDomain):
	this_output = basic_output
	this_output[zero_locations[out_i//2]] = 1
	this_output[one_locations[out_i//2]] = 0
	output_template[ctx_i*nItemsPerDomain+out_i] = this_output 
	

#Fill output matrix
for dom_i in xrange(nDomains):
    for item_i in range(nItemsPerDomain):
	for ctx_i in xrange(nContextPerDomain):
	    output_offset = dom_i*nContextPerDomain*nSingleOutputSize+ctx_i*nSingleOutputSize
	    OutputMatrix[dom_i*nContextPerDomain*nItemsPerDomain+item_i*nContextPerDomain+ctx_i,output_offset:output_offset+nSingleOutputSize] = output_template[ctx_i*nItemsPerDomain+item_i]
	    OutputMatrix_ContextCollapsed[dom_i*nItemsPerDomain+item_i,output_offset:output_offset+nSingleOutputSize] = output_template[ctx_i*nItemsPerDomain+item_i]

print OutputMatrix.shape #order: in order of input, domain -> item -> context 
plt.imshow(OutputMatrix,interpolation='none')
plt.savefig('results/OutputMatrix.png')
plt.close()
#plt.show()
#
print OutputMatrix_ContextCollapsed.shape
plt.imshow(OutputMatrix_ContextCollapsed,interpolation='none')
plt.savefig('results/OutputMatrix_ContextCollapsed.png')
plt.close()
#plt.show()
#
#Calculate similarity matrices
OutputSimilarityMatrix = numpy.dot(OutputMatrix,OutputMatrix.transpose())
#print OutputSimilarityMatrix.shape
#
plt.imshow(OutputSimilarityMatrix,cmap='Greys',interpolation='none')
plt.savefig('results/OutputSimilarityMatrix.png')
plt.close()
#plt.show()

#Output similarity matrix context collapsed
Output_ContextCollapsed_SimilarityMatrix = numpy.dot(OutputMatrix_ContextCollapsed,OutputMatrix_ContextCollapsed.transpose())
#print Output_ContextCollapsed_SimilarityMatrix.shape
#
plt.imshow(Output_ContextCollapsed_SimilarityMatrix,cmap='Greys',interpolation='none')
plt.savefig('results/Output_ContextCollapsed_SimilarityMatrix.png')
plt.close()
#plt.show()


#set up network below
inputvec_item = tf.placeholder(tf.float32, shape = [nSizeofItemInput,1])#input item vector
inputvec_context = tf.placeholder(tf.float32, shape = [nSizeofContextInput,1])#input context vector
outputvec = tf.placeholder(tf.float32, shape = [nSizeofOutput,1])#output vector

W1_item = tf.Variable(tf.random_uniform([sizeItemRepLayer,nSizeofItemInput],-1,1)) #weights from item input to item hidden layer1
B1_item = tf.Variable(tf.random_uniform([sizeItemRepLayer,1],-1,1)) #bias from item input to item hidden layer1

W1_context = tf.Variable(tf.random_uniform([sizeContextRepLayer,nSizeofContextInput],-1,1)) #weights from context input to context hidden layer1
B1_context = tf.Variable(tf.random_uniform([sizeContextRepLayer,1],-1,1)) #bias from context input to context hidden layer1

W2_item = tf.Variable(tf.random_uniform([sizeSecondLayer,sizeItemRepLayer],-1,1)) #weights from item hidden layer1 to layer2
B2_item = tf.Variable(tf.random_uniform([sizeSecondLayer,1],-1,1)) #bias from item hidden layer1 to layer2

W2_context = tf.Variable(tf.random_uniform([sizeSecondLayer,sizeContextRepLayer],-1,1)) #weights from context hidden layer1 to layer2
B2_context = tf.Variable(tf.random_uniform([sizeSecondLayer,1],-1,1)) #bias from context hidden layer1 to layer2

W3 = tf.Variable(tf.random_uniform([nSizeofOutput,sizeSecondLayer],-1,1)) #weights from context hidden layer1 to layer2
B3 = tf.Variable(tf.random_uniform([nSizeofOutput,1],-1,1)) #bias from context hidden layer1 to layer2

#construct network
itemhidden = tf.nn.sigmoid(tf.matmul(W1_item,inputvec_item) + B1_item)
contexthidden = tf.nn.sigmoid(tf.matmul(W1_context,inputvec_context) + B1_context)

secondhidden = tf.nn.sigmoid((tf.matmul(W2_item,itemhidden) + B2_item) + (tf.matmul(W2_context,contexthidden) + B2_context))

output = tf.nn.sigmoid(tf.matmul(W3, secondhidden) + B3)

error = tf.reduce_sum(tf.square(output - outputvec))
train = tf.train.AdamOptimizer(eta).minimize(error)


listerror = []

def train_error():
    train_error = 0.0
    for i in range(nNumberOfPossibleInputPatterns):
	    train_error += session.run(error, feed_dict = {inputvec_item: InputMatrix[i][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]),	
				inputvec_context: InputMatrix[i][nSizeofItemInput:].reshape([nSizeofContextInput,1]), 
				outputvec: OutputMatrix[i].reshape([nSizeofOutput,1])})
    return train_error/nNumberOfPossibleInputPatterns 

def get_network_outputs():
    network_outputs = numpy.zeros([nNumberOfPossibleInputPatterns,nSizeofOutput])
    for i in range(nNumberOfPossibleInputPatterns):
	    network_outputs[i,:] = session.run(output, feed_dict = {inputvec_item: InputMatrix[i][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]),	
				inputvec_context: InputMatrix[i][nSizeofItemInput:].reshape([nSizeofContextInput,1]), 
				outputvec: OutputMatrix[i].reshape([nSizeofOutput,1])}).flatten()
    return network_outputs


def get_item_reps():
    item_reps = numpy.zeros([nNumberOfPossibleInputPatterns,sizeItemRepLayer])
    for i in range(nNumberOfPossibleInputPatterns):
	    item_reps[i,:] = session.run(itemhidden, feed_dict = {inputvec_item: InputMatrix[i][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]),	
				inputvec_context: InputMatrix[i][nSizeofItemInput:].reshape([nSizeofContextInput,1]), 
				outputvec: OutputMatrix[i].reshape([nSizeofOutput,1])}).flatten()
    return item_reps

def get_second_reps():
    second_reps = numpy.zeros([nNumberOfPossibleInputPatterns,sizeSecondLayer])
    for i in range(nNumberOfPossibleInputPatterns):
	    second_reps[i,:] = session.run(secondhidden, feed_dict = {inputvec_item: InputMatrix[i][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]),	
				inputvec_context: InputMatrix[i][nSizeofItemInput:].reshape([nSizeofContextInput,1]), 
				outputvec: OutputMatrix[i].reshape([nSizeofOutput,1])}).flatten()
    return second_reps

def log_images(epoch=0):
	network_outputs = get_network_outputs() 
	
	plt.imshow(numpy.dot(network_outputs,OutputMatrix.transpose()),cmap='Greys',interpolation='none')
	plt.savefig('results/epoch_%i_networks_output_vs_output.png' %epoch)
	plt.close()
	
#	item_reps = get_item_reps() 
#	item_rep_norms = numpy.linalg.norm(item_reps,axis=1)
#	item_reps = item_reps / item_rep_norms[:,numpy.newaxis]
#	plt.imshow(numpy.dot(item_reps,item_reps.transpose()),cmap='Greys',interpolation='none') #cosine distance
	item_reps = get_item_reps() 
	item_rep_sim = numpy.zeros([nNumberOfPossibleInputPatterns,nNumberOfPossibleInputPatterns])
	for i in xrange(nNumberOfPossibleInputPatterns):
	    for j in xrange(i,nNumberOfPossibleInputPatterns):
		item_rep_sim[i,j] = numpy.linalg.norm(item_reps[i]-item_reps[j]) 
		item_rep_sim[j,i] = item_rep_sim[i,j]
	plt.imshow(item_rep_sim,cmap='Greys_r',interpolation='none') #cosine distance
	plt.savefig('results/epoch_%i_item_rep_similarity.png' %epoch)
	plt.close()

#	second_reps = get_second_reps() 
#	second_rep_norms = numpy.linalg.norm(second_reps,axis=1)
#	second_reps = second_reps / second_rep_norms[:,numpy.newaxis]
#	plt.imshow(numpy.dot(second_reps,second_reps.transpose()),cmap='Greys',interpolation='none') #cosine distance
	second_reps = get_second_reps() 
	second_rep_sim = numpy.zeros([nNumberOfPossibleInputPatterns,nNumberOfPossibleInputPatterns])
	for i in xrange(nNumberOfPossibleInputPatterns):
	    for j in xrange(i,nNumberOfPossibleInputPatterns):
		second_rep_sim[i,j] = numpy.linalg.norm(second_reps[i]-second_reps[j]) 
		second_rep_sim[j,i] = second_rep_sim[i,j]
	plt.imshow(second_rep_sim,cmap='Greys_r',interpolation='none') #cosine distance
	plt.savefig('results/epoch_%i_second_rep_similarity.png' %epoch)
	plt.close()

#run the network
init = tf.initialize_all_variables()
with tf.Session() as session:
	session.run(init)
	print "Initial training MSE: ",train_error()
	listerror.append(train_error())
	log_images(0)
	for epoch in xrange(1,nEpochs+1):
	    this_order = numpy.random.permutation(nNumberOfPossibleInputPatterns) #Present examples in a random order each time
	    for i in this_order:
		session.run([train, error], feed_dict = {inputvec_item: InputMatrix[i%nNumberOfPossibleInputPatterns][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]), 
															inputvec_context: InputMatrix[i%nNumberOfPossibleInputPatterns][nSizeofItemInput:].reshape([nSizeofContextInput,1]),
															outputvec: OutputMatrix[i%nNumberOfPossibleInputPatterns].reshape([nSizeofOutput,1])})
		
	    listerror.append(train_error())
	    if epoch % 10 == 0:
		print "On epoch %i, training MSE: %f" %(epoch,listerror[-1])
		i = nNumberOfPossibleInputPatterns-1
	    if epoch % 20 == 0:
		log_images(epoch)

plt.plot(range(nEpochs+1),listerror)
plt.show()
