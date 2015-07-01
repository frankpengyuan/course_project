import random
fin = open("fea/train_fea1",'r')
fo1 = open("fea/train_fea1_1",'w')
fo2 = open("fea/train_fea1_2",'w')
fin_2 = open("fea/train_fea2",'r')
fo1_2 = open("fea/train_fea2_1",'w')
fo2_2 = open("fea/train_fea2_2",'w')

for line in fin:
	if random.uniform(0,1)<0.75:
		fo1.write(line)
	else:
		fo2.write(line)

for line in fin_2:
	if random.uniform(0,1)<0.75:
		fo1_2.write(line)
	else:
		fo2_2.write(line)

fin.close()
fo1.close()
fo2.close()
fin_2.close()
fo1_2.close()
fo2_2.close()