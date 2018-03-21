"""
*************
Tree Conv based on Chow Liu Tree
*************
"""

__author__ = """Xiaopeng LI <xlibo@connect.ust.hk>"""

from collections import deque
from .chow_liu import mst_kruskal

def tree_conv(mst, root, kernel_size=1, stride=1):
	"""
	Argument
	------------
	mst: undirected maximum spanning tree
		dict, key is the vertex, value is adj list
	root: selected root

	Note
	-----------
	Implemented through bfs
	"""
	nv = len(mst)
	visited = [False]*nv
	q = deque()
	q.append(root)
	visited[root] = True
	islands = dict()
	while q:
		vertex = q.popleft()
		neighborVisited = [False]*nv
		neighborVisited[vertex] = True
		neighbors = getNeibourhood(mst, vertex, kernel_size, neighborVisited)
		islands[vertex] = [vertex] + neighbors

		# enqueue strided vertex
		neighbors = getStrideNeighbourhood(mst, vertex, stride, visited)
		for x in neighbors:
			q.append(x)

	return islands

def getNeibourhood(mst, root, size, neighborVisited):
	if size==0:
		return []
	neighbors = []
	for v in mst[root]:
		if not neighborVisited[v]:
			neighbors.append(v)
			neighborVisited[v] = True
			neighbors += getNeibourhood(mst, v, size-1, neighborVisited)
	return neighbors

def getStrideNeighbourhood(mst, root, size, visited):
	if size==0:
		return [root]
	neighbors = []
	for v in mst[root]:
		if not visited[v]:
			visited[v] = True
			neighbors += getStrideNeighbourhood(mst, v, size-1, visited)
	return neighbors

if __name__ == "__main__":
	mst = dict()
	mst[0] = [1,2]
	mst[1] = [0, 3, 4]
	mst[2] = [0,5,6]
	mst[3] = [1,7,8]
	mst[4] = [1,9]
	mst[5] = [2]
	mst[6] = [2]
	mst[7] = [3]
	mst[8] = [3]
	mst[9] = [4]
	islands = tree_conv(mst, 0, kernel_size=2, stride=2)
	for e in islands:
		print(e, islands[e])
