import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

#set properties of neural net such as size, input, output
nDomains = 4
nItemsPerDomain = 8
nContextPerDomain = 4
nSingleOutputSize = 14

nSizeofItemInput = nDomains*nItemsPerDomain #size of item input vector
nSizeofContextInput = nDomains*nContextPerDomain#size of context input vector

nSizeofInput = (nItemsPerDomain+nContextPerDomain)*nDomains #length of a single input vector
nSizeofOutput = nSingleOutputSize*nDomains*nContextPerDomain # length of a single output vector
nNumberOfInputs = nItemsPerDomain*nContextPerDomain*nDomains #number of training examples in dataset

sizeItemRepLayer = 32
sizeContextRepLayer = 16
sizeSecondLayer = 48

#construct the input matrix
InputMatrix = numpy.empty([nNumberOfInputs,nSizeofInput],int)
InputCount = 0

A = numpy.zeros((nItemsPerDomain,nDomains),dtype =numpy.int)
B = numpy.zeros((nContextPerDomain,nDomains),dtype =numpy.int)


for i in range(nDomains):
	for j in range(nItemsPerDomain):
		A[j,i] = 1
		for k in range(nContextPerDomain):
			B[k,i] = 1
			tempA = A.flatten('F')
			tempB = B.flatten('F')
			temp = numpy.concatenate((tempA,tempB))
			InputMatrix[InputCount] = temp
			InputCount = InputCount + 1

			B = numpy.zeros((nContextPerDomain,nDomains))
		A = numpy.zeros((nItemsPerDomain,nDomains))
	

print InputMatrix.shape #each row is an input vector, number of row is number of training example

#construct the output matrix
OutputMatrix = numpy.empty([nNumberOfInputs,nSizeofOutput],int)
OutputCount = 0

C = numpy.zeros((nContextPerDomain*nDomains,nSingleOutputSize), dtype =numpy.int)
temprand = numpy.random.randint(2,size=(nDomains*nItemsPerDomain,nSingleOutputSize))


for a in range(nDomains*nContextPerDomain):
	for b in range(nItemsPerDomain):
		C[a] = temprand[(b + (a*nItemsPerDomain))%32] #loop over random matrix again to have same output among domains

		tempC = C.flatten()
		OutputMatrix[OutputCount] = tempC
		OutputCount = OutputCount + 1

		C = numpy.zeros((nContextPerDomain*nDomains,nSingleOutputSize), dtype =numpy.int)


print OutputMatrix.shape #order: in order of input, first fill all items in domain #1, context #1 -> fill all items in domain #1, context #2; move to domain #2 once all of domain 1 filled


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
itemhidden = tf.nn.relu6(tf.matmul(W1_item,inputvec_item) + B1_item)
contexthidden = tf.nn.relu6(tf.matmul(W1_context,inputvec_context) + B1_context)

secondhidden = tf.nn.relu6((tf.matmul(W2_item,itemhidden) + B2_item) + (tf.matmul(W2_context,contexthidden) + B2_context))

output = tf.nn.sigmoid(tf.matmul(W3, secondhidden) + B3)

error = tf.reduce_sum(tf.square(output - outputvec))
train = tf.train.GradientDescentOptimizer(0.05).minimize(error)


listerror = []
#run the network
model = tf.initialize_all_variables()
with tf.Session() as session:
	session.run(model)
	for i in range(10000):
		_ , outerror = session.run([train, error], feed_dict = {inputvec_item: InputMatrix[i%nNumberOfInputs][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]), 
															inputvec_context: InputMatrix[i%nNumberOfInputs][nSizeofItemInput:].reshape([nSizeofContextInput,1]),
															outputvec: OutputMatrix[i%nNumberOfInputs].reshape([nSizeofOutput,1])})
		
		if i%1 == 0:
			listerror.append(outerror)

plt.plot([numpy.mean(listerror[k-nNumberOfInputs:k]) for k in range(len(listerror))])
plt.show()
