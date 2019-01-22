import pickle as p

with open('speakerid2party.pkl', 'rb') as file:
	speakerid2party = p.load(file)
	for k in speakerid2party.keys():
		print(k)
	print()
