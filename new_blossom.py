import networkx as nx 
import numpy as np
from Env import Env
import matplotlib.pyplot as plt
import copy

	
def create(observation):
	G = nx.Graph()
	for i in range(observation.shape[2]):
		state = observation[:,:,i]
		errors = getErrorIndices(state)
		middleIndex = int(np.floor(state.shape[0]/2))
		centerError = int(state[middleIndex,middleIndex])
		for error in errors:
			xdist = np.abs(error[0] - middleIndex)
			ydist = np.abs(error[1] - middleIndex)
			
			errorDistance = np.abs(xdist+ydist)
			errorLabel = int(state[error[0],error[1]])
			
			if errorDistance > 0:
				G.add_edge(centerError,errorLabel, weight = errorDistance)
	return G
	
	
	
	
def printGraph(G):
	
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos, node_size=700)
	nx.draw_networkx_edges(G, pos)
	labels = nx.get_edge_attributes(G,'weight')
	nx.draw_networkx_edge_labels(G,pos,font_size=20,edge_labels=labels)
	nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
	plt.show()
	
def getErrorIndices(state):
	return np.transpose(np.nonzero(state))



def find_maximum_matching(G,M):
	P = finding_aug_path(G,M)
	if P == []:#Base Case
		return M
	else: #Augment P to M
		##Add the alternating edges of P to M
		for i in range(0,len(P)-2,2): 
			M.add_edge(P[i],P[i+1])
			M.remove_edge(P[i+1],P[i+2])
		M.add_edge(P[-2],P[-1])
		return find_maximum_matching(G,M)

def dist_to_root(point,root,Graph):
	path = nx.shortest_path(Graph, source = point, target = root)
	return (len(path)-1)
	
def finding_aug_path(G,M, Blossom_stack=[]):
	Forest = [] #Storing the Forests
	Path = [] # The final path 

	unmarked_edges = list(set(G.edges())- set(M.edges()))  
	unmarked_nodes = list(G.nodes())
	Forest_nodes = []
	## we need a map from v to the tree
	tree_to_root = {} # key=idx of tree in forest, val=root
	root_to_tree = {} # key=root, val=idx of tree in forest
		
	##List of exposed vertices - ROOTS OF TREES
	exp_vertex = list(set(G.nodes()) - set(M.nodes()))
	
	counter = 0
	#List of trees with the exposed vertices as the roots
	for v in exp_vertex:  
		temp = nx.Graph()
		temp.add_node(v)
		Forest.append(temp)
		Forest_nodes.append(v)

		#link each root to its tree
		tree_to_root[counter] = v
		root_to_tree[v] = counter
		counter = counter + 1

	
	for v in Forest_nodes:  
		root_of_v = None
		tree_num_of_v = None
		for tree_number in range(len(Forest)): 
			tree_in = Forest[tree_number]
			if tree_in.has_node(v) == True:
				root_of_v = tree_to_root[tree_number]
				tree_num_of_v = tree_number
				break #Break out of the for loop
		edges_v = list(G.edges(v))
		for edge_number in range(len(edges_v)): 
			e = edges_v[edge_number]
			e2 = (e[1],e[0]) #the edge in the other order
			if ((e in unmarked_edges or e2 in unmarked_edges) and e!=[]):
				w = e[1] # the other vertex of the unmarked edge
				w_in_Forest = 0; ##Indicator for w in F or not

				##Go through all the trees in the forest to check if w in F
				tree_of_w = None
				tree_num_of_w = None
				for tree_number in range(len(Forest)):
					tree = Forest[tree_number]
					if tree.has_node(w) == True:
						w_in_Forest = 1
						root_of_w = tree_to_root[tree_number]
						tree_num_of_w = tree_number
						tree_of_w = Forest[tree_num_of_w]
						break #Break the outer for loop
				
				if w_in_Forest == 0:
					## w is matched, so add e and w's matched edge to F
					Forest[tree_num_of_v].add_edge(e[0],e[1]) # edge {v,w}
					# Note: we don't add w to forest nodes b/c it's odd dist from root
					#assert(M.has_node(w))
					edge_w = list(M.edges(w))[0] # get edge {w,x}
					Forest[tree_num_of_v].add_edge(edge_w[0],edge_w[1]) # add edge{w,x}
					Forest_nodes.append(edge_w[1]) ## add {x} to the list of forest nodes

				else: ## w is in Forest
					# if odd, do nothing.
					if dist_to_root(w,root_of_w,Forest[tree_num_of_w])%2 == 0:
						if (tree_num_of_v != tree_num_of_w):
							##Shortest path from root(v)--->v-->w---->root(w)
							path_in_v = nx.shortest_path(Forest[tree_num_of_v], source = root_of_v, target = v)
							path_in_w = nx.shortest_path(Forest[tree_num_of_w], source = w, target = root_of_w)

							return path_in_v + path_in_w
						else: ##Contract the blossom
							# create blossom
							blossom = nx.shortest_path(tree_of_w, source=v, target=w)
							blossom.append(v)
							#assert(len(blossom)%2 == 0)
							# contract blossom into single node w
							contracted_G = copy.deepcopy(G)
							contracted_M = copy.deepcopy(M)
							for node in blossom[0:len(blossom)-1]:
								if node != w:
									contracted_G = nx.contracted_nodes(contracted_G, w, node, self_loops=False)
									if node in contracted_M.nodes(): 
									   edge_rm = list(M.edges(node))[0] #this will be exactly one edge
									   contracted_M.remove_node(node)
									   contracted_M.remove_node(edge_rm[1])
									   #assert(len(list(contracted_M.nodes()))%2 == 0)
							# add blossom to our stack
							Blossom_stack.append(w)

							# recurse
							aug_path = finding_aug_path(contracted_G, contracted_M, Blossom_stack)

							# check if blossom exists in aug_path 
							v_B = Blossom_stack.pop()
							if (v_B in aug_path):
								##Define the L_stem and R_stem
								L_stem = aug_path[0:aug_path.index(v_B)]
								R_stem = aug_path[aug_path.index(v_B)+1:]
								lifted_blossom = [] #stores the path within the blossom to take
								# Find base of blossom
								i = 0
								base = None
								base_idx = -1
								blossom_ext = blossom + [blossom[1]] 
								while base == None and i < len(blossom) - 1:
									if not(M.has_edge(blossom[i],blossom[i+1])):
										if not(M.has_edge(blossom[i+1],blossom_ext[i+2])): 
											base = blossom[i+1]
											base_idx = i+1
										else:
											i += 2
									else:
										i += 1
								# if needed, create list of blossom nodes starting at base
								if blossom[0] != base:
									based_blossom = []
									base_idx = blossom.index(base)
									for i in xrange(base_idx,len(blossom)-1):
										based_blossom.append(blossom[i])
									for i in xrange(0,base_idx):
										based_blossom.append(blossom[i])
									based_blossom.append(base)
								else:
									based_blossom = blossom

								# CHECK IF BLOSSOM IS ENDPT
								if L_stem == [] or R_stem == []:
									if L_stem != []:
										if G.has_edge(base, L_stem[-1]):
											# CASE 1:
											# Chuck the blossom
											return L_stem + [base]
										else:
											# CASE 2:
											# find where Lstem is connected
											i = 1
											while (lifted_blossom == []):
												#assert(i < len(based_blossom)-1)
												if G.has_edge(based_blossom[i],L_stem[-1]):
													# make sure we're adding the even part to lifted path
													if i%2 == 0: # same dir path
														lifted_blossom = list(reversed(based_blossom))[-i-1:] ####################
													else: # opposite dir path
														lifted_blossom = based_blossom[i:]##########################
												i += 1
											return L_stem + lifted_blossom

									else:
										if G.has_edge(base, R_stem[0]):
											# CASE 1:
											# Chuck the blossom. 
											return [base] + R_stem
										else:
											# CASE 2:
											# find where R_stem is connected
											i = 1
											while (lifted_blossom == []):
												#assert(i < len(based_blossom)-1)
												if G.has_edge(based_blossom[i],R_stem[0]):
													# make sure we're adding the even part to lifted path
													if i%2 == 0: # same dir path
														lifted_blossom = based_blossom[:i+1]
														#print lifted_blossom
													else: # opposite dir path
														lifted_blossom = list(reversed(based_blossom))[:-i]
												i += 1
											return lifted_blossom + R_stem

								else: # blossom is in the middle
									# LIFT the blossom
									# check if L_stem attaches to base
									if M.has_edge(base, L_stem[-1]):
										# find where right stem attaches
										if G.has_edge(base, R_stem[0]):
											# blossom is useless
											return L_stem + [base] + R_stem
										else:
											# blossom needs to be lifted
											i = 1
											while (lifted_blossom == []):
												# assert(i < len(based_blossom)-1)
												if G.has_edge(based_blossom[i],R_stem[0]):
													# make sure we're adding the even part to lifted path
													if i%2 == 0: # same dir path
														lifted_blossom = based_blossom[:i+1] 
														# print lifted_blossom
													else: # opposite dir path
														lifted_blossom = list(reversed(based_blossom))[:-i]
														# print lifted_blossom
												i += 1
											return L_stem + lifted_blossom + R_stem
									else: 
										# R stem to base is in matching
										# assert(M.has_edge(base, R_stem[0]))
										# check where left stem attaches
										if G.has_edge(base, L_stem[-1]):
											# blossom is useless
											return L_stem + [base] + R_stem
										else:
											# blossom needs to be lifted
											i = 1
											while (lifted_blossom == []):
												# assert(i < len(based_blossom)-1)
												if G.has_edge(based_blossom[i],L_stem[-1]):
													# make sure we're adding the even part to lifted path
													if i%2 == 0: # same dir path
														lifted_blossom = list(reversed(based_blossom))[-i-1:] 
													else: # opposite dir path
														lifted_blossom = based_blossom[i:] 
												i += 1
											return L_stem + list((lifted_blossom)) + R_stem
							else: # blossom is not in aug_path
								return aug_path
	##IF Nothing is Found
	return Path ##Empty Path


if __name__ == '__main__':
	A = np.zeros((5,5))
	
	A[1,0] = 3
	A[1,1] = 4
	A[0,1] = 1
	A[0,2] = 2
	A[2,1] = 5
	A[2,2] = 6
	A[0,3] = 7
	A[0,4] = 8
	print(A)
	env = Env(A)
	obs = env.getObservation()
	G = create(obs)
	M = nx.Graph()
	MM = find_maximum_matching(G, M)
	printGraph(G)
	printGraph(MM)
