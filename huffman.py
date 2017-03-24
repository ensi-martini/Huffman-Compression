"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode
import cProfile
import pstats


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    
    output_dict = {}
    
    for t in text:
    
        #If this byte isn't in the dict, add it
        if t not in output_dict:
            output_dict[t] = 0
            
        output_dict[t] += 1
        
    return output_dict  

def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """    
    
    #We need to treat 1 item in the dict as a special case
    if len(freq_dict) >= 2:
        
        node_list = []
        
        #Add every key and value in the freq dict as a HNode with their proper
        #number values
        for n in freq_dict:

            #Letters that do not appear but are in the dict do not get added
            if freq_dict[n] > 0:
                
                node = HuffmanNode(n)
                node.number = freq_dict[n] #We don't use a cache - unique keys
                node_list.append(node)
                
        #Sort the nodes by their number attribute
        node_list.sort(key=lambda x: x.number)
        
        #After sorting from smallest to greatest occurances, iterate each time
        #and merge the two least frequent nodes together (could be internals)
        #in an effort to build the list from the bottom up
        
        #This keeps going until only one item is left (the root)
        while len(node_list) > 1:
            
            #Create a node with children being two least frequent nodes                           
            current_node = HuffmanNode(None, node_list[0], node_list[1])
            
            #Set it's number value to be the sum of it's children
            current_node.number = node_list[0].number + node_list[1].number
            
            #Now remove those two children from the consideration list
            #(they are represented as a culmulative unit under the new node)
            node_list = node_list[2:] + [current_node]
            
            #Now we re-sort with our new node
            node_list.sort(key=lambda x: x.number)
            
        #Once we have built the tree all the way up, we can return it's root
        return node_list[0]
        
    #Special case - only one item in the dict
        
    #Create a empty root and set it's left to have a symbol and value of the
    #only key and value in the dict, respectively
    root = HuffmanNode(None)
    left = HuffmanNode(max(freq_dict))
    left.number = freq_dict[left.symbol]
    root.left = left
    
    return root
    
def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    
    def _get_codes(t, n):
        ''' Find n in t and return it's binary address, or empty string if it is
        not found
        '''
        
        
        address = ''
        
        if t:
            
            #If the symbol we are looking for is on the left
            if t.left and t.left.symbol == n:
                return '0'
            
            #If it is on the right
            elif t.right and t.right.symbol == n:
                return '1'
            
            #If we have a left but it's not our symbol
            if t.left:
                #Parse the children of the left node and check them for the same
                #symbol
                temp = _get_codes(t.left, n) 
                
                #If it finds it, it will have a binary value and we will know
                #to add all the bits for the directions there from the root
                if temp != '':
                    address += '0' + temp
            
            #If we have a right but it's not our symbol
            if t.right:
                #Parse the children of the left node and check them for the same
                #symbol                
                temp = _get_codes(t.right, n) 
                
                #If it finds it, it will have a binary value and we will know
                #to add all the bits for the directions there from the root                
                if temp != '':
                    address += '1' + temp             
            
        return address
    
    output = {}
    
    
    #Simply parse the tree and run the helper function on each node
    if tree:
        
        if tree.left:
            
            if tree.left.is_leaf(): #When we find a leaf, get that symbol's code
                output[tree.left.symbol] = _get_codes(tree, tree.left.symbol)

            else:
                #If its internal, repeat on the left subtree
                temp = get_codes(tree.left)
                                
                for t in temp: #Add leading 0 to take into account current level
                    temp[t] = '0' + temp[t]
                    
                #Add all newly-created keys and values to our dict
                #We don't have to worry about overwrites, each leaf is unique
                output.update(temp)
            
        if tree.right: 
        
            if tree.right.is_leaf(): #If we find a leaf, get that symbol's code
                output[tree.right.symbol] = _get_codes(tree, tree.right.symbol)
                
            else:
                #If its internal, repeat on the right subtree
                temp = get_codes(tree.right)
                
                for t in temp: #Add leading 1 to take into account current level
                    temp[t] = '1' + temp[t]
                    
                #Add all newly-created keys and values to our dict
                #We don't have to worry about overwrites, each leaf is unique                
                output.update(temp) 
                
    return output


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    
    def _internal_post(tree):
        ''' Return a list of internal nodes, found in postorder
        '''
        
        #if we hit a leaf node         
        if not tree or tree.is_leaf():
            return []
        
        #Similar to postorder parsing in class, but return list of HNodes  
        #themselves instead of simply their symbols
        return _internal_post(tree.left) + _internal_post(tree.right) + [tree]
            
    #Get the postorder list from the helper function
    internals = _internal_post(tree)
    
    for i in range(len(internals)):
        #We already have our list in postorder, now just set their number 
        #attribute to match their index
        internals[i].number = i


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    
    total = 0
    #Divisor is total occurrences of all values
    divisor = sum(freq_dict.values())
    
    code_dict = get_codes(tree) #Dict values give us the weight of each key
    
    for k in code_dict:
        #Add it based on the appropriate weight
        total += len(code_dict[k]) * freq_dict[k]
        
    return total / divisor

def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    #SIMPLIFY
    lines = []    
    cache = {}

    for t in text:
        lines.append(codes[t])
        
    compressed = ''.join(lines)
    length = len(compressed)
    remainder = 0
    
    if length % 8 != 0:
        remainder = ( length % 8)  * -1
        remaining = compressed[remainder:]
        compressed = compressed[:remainder]
    
    output = []
    
    for i in range(0, length + remainder, 8):
        current = compressed[i:i + 8]
        
        if current not in cache:
            cache[current] = bits_to_byte(current)
        output.append(cache[current])

    if length % 8 != 0:
        
        remaining = '{:0<8s}'.format(remaining)
        
        if remaining in cache:
            output.append(cache[remaining])
            
        else:
            output.append(bits_to_byte('{:0<8s}'.format(remaining)))

    return bytes(output)

def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    
    output = []
    
    def _tree_to_bytes(tree):
        '''
        Return a postorder list of None nodes in tree
        
        '''
        
        #If we dont have a tree our we have a leaf node
        if not tree or tree.is_leaf():
            return []
        
        #Return a list of the postorder traversal of the nodes
        return _tree_to_bytes(tree.left) + _tree_to_bytes(tree.right) + [tree]
    
    #Create a list that gives us a postorder traversal of the None nodes within
    #our tree
    ordered = _tree_to_bytes(tree)
    
    #For each None node in our ordered list
    for o in ordered:
        
        
        #If our left exists and is a leaf node, append a 0, 
        #then append its symbol
        if o.left and o.left.is_leaf():
            output.append(0)
            output.append(o.left.symbol)
         
        #Else if it is just an internal node, append a 1 and then the number
        #that internal node has - its postorder number
        elif o.left:
            output.append(1)
            output.append(o.left.number)
            
        #If our right exists and is a leaf node, append a 0, 
        #then append its symbol        
        if o.right and o.right.is_leaf():
            output.append(0)
            output.append(o.right.symbol)
            
        #Else if it is just an internal node, append a 1 and then the number
        #that internal node has - its postorder number    
        elif o.right:
            output.append(1)
            output.append(o.right.number)
            
    #Return a bytes object of our output list
    return bytes(output)
    
def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)

    tree = huffman_tree(freq)

    codes = get_codes(tree)

    number_nodes(tree)

    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
             size_to_bytes(len(text)))
    result += generate_compressed(text, codes)

    with open(out_file, "wb") as f2:
        f2.write(result)



# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """

    #Create a copy of the node list
    nodes = node_lst[:]
    
    #Make our root, the root index we were given
    root = nodes.pop(root_index)
    
    #We know that each node has it's node number at the same place as it's
    #index, so if our root is not the last item in the list we need to
    #put some placeholder to compensate
    
    #At our root_index, replace it with None, so as it doesnt change the length
    #of the list
    nodes.insert(root_index, None)
    
    #Create a new HuffmanNode to be our root node
    root_node = HuffmanNode()

    #If the left type of our original root is an internal node
    if root.l_type == 1:
        #With our root_node, recurse through the tree except our root index
        #is now the data of our left internal node
        root_node.left = generate_tree_general(nodes, root.l_data)
    
    #Else make the left of our root_node a new Node with left data of root
    else:
        root_node.left = HuffmanNode(root.l_data)
        
    #Same process as with left except with right
    if root.r_type == 1:
        root_node.right = generate_tree_general(nodes, root.r_data)
    
    else:
        root_node.right = HuffmanNode(root.r_data)
        
    return root_node

def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
        """
    
    def _count_internals(t):
        '''
        Return the total amount of internal nodes in tree t.
        
        '''
        
        total = 0
        
        #If we have a tree
        if t:
            
            #If t.left is an internal node, add 1 to the total and repeat
            #With the left subtree
            if t.left and not t.left.is_leaf():
                total += _count_internals(t.left) + 1
                
            #If t.right is an internal node add 1 to the total and repeat
            #With the right subtree
            if t.right and not t.right.is_leaf():
                total += _count_internals(t.right) + 1
                
        return total    
    
    #Create a copy of the nodes list
    nodes = node_lst[:]
    
    #Make our root the node at the given index for nodes
    root = nodes[root_index]
    
    #We know that each index is the postorder equivalent of the parent at that 
    #subtree
    
    #We also know that the root's right child, if it is a subtree, has a root 
    #with a postorder value of root_index - 1
    
    #The root node will always be the last one, in postorder
    
    #We cannot use the l_data or r_data properties for internals
    
    root_node = HuffmanNode()
    
    #If the right tree is an internal node
    if root.r_type == 1:
        
        #generate the post order with a smaller list, and a shifted root_index
        root_node.right = generate_tree_postorder(nodes[:-1], root_index - 1)
        
    else:
        #make the right of the root node with the data given
        root_node.right = HuffmanNode(root.r_data)
        
    #After we have reached the end of the right side, the 
    #next found one on the left will be the one before in the postorder repr
    
    
    #Repeat the same process but with the left data and subtrees
    if root.l_type == 1:
        
        internals = _count_internals(root_node)
        root_node.left = generate_tree_postorder(nodes[:-1 * internals], \
                                                 root_index - 1 - internals)
        
    else:
        root_node.left = HuffmanNode(root.l_data)
        
    return root_node
        

            
def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompressd
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    
    #We take a tree, compressed bytes like the output in our generate_compressed
    #and a number of items to decode, to avoid adding padding to output
    
    #get the codes of the given tree
    codes = get_codes(tree)

    #inverse the dictionary of codes we created
    inverse_codes = {value: key for key, value in codes.items()}
    
    bit_form = []
    #Use a cache to not have to recalculte byte_to_bits if we already know it
    #for any value
    cache = {}
    
    for b in text:
        
        #We get all of our strings as list items, runtime is faster than simply
        #adding strings each iteration        
    
        if b not in cache:
            cache[b] = byte_to_bits(b)
            
        bit_form.append(cache[b])
        
    #Now we can do one simple join to concatanate to a single string
    bit_form = ''.join(bit_form)
    
    output = []
    
    #Another time saver: instead of actually slicing a string to compare, we can
    #just check between the indicies without equating to anything
    check_from = 0
    check_to = 1
    found = 0
    
    while found < size:
    
        checker = bit_form[check_from:check_to]
        
        if checker in inverse_codes:
            #Once we find a match, we can go ahead and add it to our output
            #and push our starting pointer
            output.append(inverse_codes[checker])
            check_from = check_to
            found += 1
            
        check_to += 1
            
    return bytes(output)

def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            something = generate_uncompressed(tree, text, size)
            g.write(something)
            
# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """                   

    info = []
    
    #We append occurrences and symbols as a tuple
    for key in freq_dict: 
        info.append((freq_dict[key], key))
    
    #Sort based on the occurrences, then make our list only the symbols
    symbols = sorted(info, reverse=True)
    symbols = [x[1] for x in symbols]
    
    nodes = [tree]
    
    #This code works on the basis of checking each level before moving down the
    #tree, WITHOUT using recursion. When we find any leaf at the next level
    #we can give it the symbol of the most frequent node that we still have not
    #assigned anywhere
    
    #When we find an internal child of the current node, we simply append it to
    #a new list for the next iteration, and we keep doing this until we have 
    #reassigned all our nodes
    
    while symbols != []:
        
        new_nodes = []
        
        for n in nodes:
                
            if n.left and n.left.is_leaf():
                n.left.symbol = symbols.pop(0)
                
            else:
                new_nodes.append(n.left)
                
            if n.right and n.right.is_leaf():
                n.right.symbol = symbols.pop(0)
                
            else:
                new_nodes.append(n.right)
                
        nodes = new_nodes[:]
        new_nodes = []
        
if __name__ == "__main__":

    #letters = {'A':3, 'B':1, 'C':4, 'D':6, 'E':2, 'F':5}
    ##print(avg_length(tree, freq))
    #tree = HuffmanNode()
    #tree.left = HuffmanNode()
    #tree.left.left = HuffmanNode('A')
    #tree.left.right = HuffmanNode()
    #tree.left.right.left = HuffmanNode('B')
    #tree.left.right.right = HuffmanNode('C')
    
    #tree.right = HuffmanNode()
    #tree.right.left = HuffmanNode('D')
    #tree.right.right = HuffmanNode()
    #tree.right.right.left = HuffmanNode('E')
    #tree.right.right.right = HuffmanNode('F')
    #t = improve_tree(tree, letters)
    #print(t.symbol)
    #print(t.left.symbol)
    #print(t.left.left.symbol == 'D')
    #print(t.left.right.symbol)
    #print(t.left.right.left.symbol == 'C')
    #print(t.left.right.right.symbol == 'A')
    #print(t.right.symbol)
    #print(t.right.left.symbol == 'F')
    #print(t.right.right.right.symbol == 'B')
    #print(t.right.right.left.symbol == 'E')
    #print(avg_length(tree, freq))
    cProfile.run('compress("a.txt", "a.huf")')
    cProfile.run('uncompress("a.huf", "one.txt")')
    
    #ht = HuffmanNode()
    #ht.left = HuffmanNode(0)
    #ht.right = HuffmanNode()
    #ht.right.left = HuffmanNode(10)
    #ht.right.right = HuffmanNode(11)
    #print(get_codes(ht))
    
    #import python_ta
    #python_ta.check_all(config="huffman_pyta.txt")
    ##TODO: Uncomment these when you have implemented all the functions
    
    #import doctest
    #doctest.testmod()

    #import time

    #mode = input("Press c to compress or u to uncompress: ")
    #if mode == "c":
        #fname = input("File to compress: ")
        #start = time.time()
        #compress(fname, fname + ".huf")
        #print("compressed {} in {} seconds."
        #.format(fname, time.time() - start))
    #elif mode == "u":
        #fname = input("File to uncompress: ")
        #start = time.time()
        #uncompress(fname, fname + ".orig")
        #print("uncompressed {} in {} seconds.".format(fname, time.time() - start))
