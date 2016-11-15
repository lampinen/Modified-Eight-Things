import numpy
import matplotlib.pyplot as plt

#read input output data from Eight things file or file with same format
#output numpy matrix of input and outputs and training set size, each row is one training example
#input vector contains both context input and item input, order is item then context
def read_data_file(filename):
	textread = numpy.genfromtxt(filename, delimiter = ',', dtype = None)

	inputvector = numpy.array(textread[:,1])
	outputvector = numpy.array(textread[:,2])
	nTrainingSize = inputvector.size

	inputtemp = []
	for j in range(nTrainingSize):
		x = [int(i) for i in inputvector[j].split()]
		inputtemp.append(x)

	inputmatrix = numpy.array(inputtemp)

	outputtemp = []
	for j in range(nTrainingSize):
		x = [int(i) for i in outputvector[j].split()]
		outputtemp.append(x)

	outputmatrix = numpy.array(outputtemp)

	#plt.imshow(inputmatrix, interpolation = 'none')
	#plt.imshow(outputmatrix, interpolation = 'none')
	#OutputSimilarityMatrix = numpy.dot(outputmatrix,outputmatrix.transpose())
	#print OutputSimilarityMatrix
	#numpy.savetxt('Output Dataset.txt', outputmatrix, fmt = '%u')
	return inputmatrix, outputmatrix, nTrainingSize
	
#take input matrix(outp) dataset and make it into multi domain dataset
def make_multi_domain(outp, nDomains):
	#inp,outp,c = read_data_file('EightThingsmod.txt')
	#print outp.shape
	rows = outp[:,0].size
	columns = outp[0].size

	outputMatrix = numpy.zeros([nDomains*outp[:,0].size,nDomains*outp[0].size],int)
	#print outputMatrix.shape

	for a in range(nDomains):
		outputMatrix[a*rows:(a+1)*rows,a*columns:(a+1)*columns] = outp

	return outputMatrix
	#numpy.savetxt('test.txt', outputMatrix, fmt = '%u')

#construct the output matrix
#output order: in order of input, 
#first fill all items in domain #1, context #1 -> fill all items in domain #1, context #2; move to domain #2 once all of domain 1 filled
def rand_output_matrix(nDomains, nContextPerDomain, nItemsPerDomain, nSingleOutputSize):
	nNumberOfPossibleInputs = nItemsPerDomain*nContextPerDomain*nDomains
	nSizeofOutput = nSingleOutputSize*nDomains*nContextPerDomain

	OutputMatrix = numpy.empty([nNumberOfPossibleInputs,nSizeofOutput],int)
	OutputCount = 0

	C = numpy.zeros((nContextPerDomain*nDomains,nSingleOutputSize), dtype =numpy.int)
	temprand = numpy.random.randint(2,size=(nDomains*nItemsPerDomain,nSingleOutputSize))


	for a in range(nDomains*nContextPerDomain):
		for b in range(nItemsPerDomain):
			C[a] = temprand[(b + (a*nItemsPerDomain))%nDomains*nItemsPerDomain] #loop over random matrix again to have same output among domains

			tempC = C.flatten()
			OutputMatrix[OutputCount] = tempC
			OutputCount = OutputCount + 1

			C = numpy.zeros((nContextPerDomain*nDomains,nSingleOutputSize), dtype =numpy.int)

	#numpy.savetxt('Output Dataset.txt', OutputMatrix, fmt = '%u')
	return OutputMatrix

#inputs: number of domains, number of context per domain, number of items per domain
#output numpy matrix each row is an input vector, number of row is number of training example, 
def input_matrix(nDomains, nContextPerDomain, nItemsPerDomain):
	nSizeofInput = (nItemsPerDomain+nContextPerDomain)*nDomains
	nNumberOfPossibleInputs = nItemsPerDomain*nContextPerDomain*nDomains

	InputMatrix = numpy.empty([nNumberOfPossibleInputs,nSizeofInput],int)
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
		
	#numpy.savetxt('Input Dataset.txt', InputMatrix, fmt = '%u')
	return InputMatrix

#euclidean distance between two numpy input vectors
def euclid_dis(a,b):
	c = numpy.sqrt(numpy.sum((a-b)**2))
	return c

#euclidean distance matrix
#input a matrix, output the euclidean distance matrix for the pairwise distance between all pairs of rows
def euclid_dis_matrix(a):
	rows = a[:,0].size
	dismatrix = numpy.empty((rows,rows))

	for x in range(rows):
		for y in range(rows):
			#dismatrix[x][y] = euclid_dis(a[x],a[y])
			dismatrix[x][y] = numpy.linalg.norm(a[x] - a[y])


	return dismatrix

#return matrix, output the dot product between all pairs of input rows
def dotproduct_matrix(a):
	rows = a[:,0].size
	dismatrix = numpy.empty((rows,rows))

	for x in range(rows):
		for y in range(rows):
			dismatrix[x][y] = numpy.dot(a[x],a[y])
	return dismatrix

#collapse output matrix a, across context for nContextPerDomain
def context_collapse(a,nContextPerDomain):
	rows = a[:,0].size
	columns = a[0].size
	collapse_matrix = numpy.empty((rows/nContextPerDomain,columns))

	for x in range(rows/nContextPerDomain):
		temp = numpy.zeros((1,columns))

		for y in range(nContextPerDomain):
			temp = temp + a[x*(nContextPerDomain) + y]

		collapse_matrix[x] = temp

	return collapse_matrix



# a,b,c = read_data_file('EightThingsshaw2.txt')
# outp = make_multi_domain(b,4)
# collp = context_collapse(outp,4)
# simimatrix = euclid_dis_matrix(collp)


#test = numpy.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])



#graph_PCA(collp)

#print collp.shape
#numpy.savetxt('test.txt', collp, fmt = '%u')
#numpy.savetxt('test2.txt', simimatrix)
#plt.imshow(simimatrix,cmap='Greys_r',interpolation='none')
# plt.imshow(simimatrix,cmap='Greys',interpolation='none')
#plt.savefig('EightThingsshaw2.png')
#plt.close()