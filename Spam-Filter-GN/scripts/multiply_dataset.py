import sys

def multiply_dataset(factor=2):
    """ Multiplies dataset by default value of 2. 
    User can set factor to any number greater than 1"""
    
    with open('data/shuffledfeats.dat', 'r') as singleFeats:
        if (factor > 1):
            newFileName = 'data/' + str(factor) + 'xshuffledfeats.dat'
            print newFileName
            with open(newFileName, 'w') as newFeats:
                for line in singleFeats:
                    for i in range(factor):
                        newFeats.write(line)

factor = int(sys.argv[1])
multiply_dataset(factor)

                                    

       

