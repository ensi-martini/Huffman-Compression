"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


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

    if len(freq_dict) >= 2:
        
        #Find and remove the most common character, without modifying the OG
        #dictionary
        temp = freq_dict.copy()
        biggest = max(temp, key=temp.get)
        biggest = [biggest, temp.pop(biggest)]
        
        if len(freq_dict) == 2:
            
            smaller = max(temp)
            smaller = [smaller, temp.pop(smaller)]
            
            left = HuffmanNode(smaller[0])
            right = HuffmanNode(biggest[0])
            left.number = smaller[1]
            right.number = biggest[1]
            
            root = HuffmanNode(None, left, right)
            root.number = smaller[1] + biggest[1]
            
        
        elif len(freq_dict) > 2:
            
            #We already got rid of the largest value
            total = sum(temp.values())
            
            #If None's total is greater than our most common character
            #Place the most common character on the left
            if total > biggest[1]:
                #We create this part first in order to set it's number attribute
                left = HuffmanNode(symbol=biggest[0])
                left.number = biggest[1]            
                right = huffman_tree(temp)
            
            else:
                
                right = HuffmanNode(symbol=biggest[0])
                right.number = biggest[1]       
                left = huffman_tree(temp)
                
                
            root = HuffmanNode(None, left, right)        
            root.number = total + biggest[1]
        
    else:
        
        last = min(freq_dict)
        child = HuffmanNode(last)
        child.number = freq_dict[last]
        root = HuffmanNode(None, child)
        root.number = child.number
        
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
    
            if t.left and t.left.symbol == n:
                return '0'
            
            elif t.right and t.right.symbol == n:
                return '1'
            
            if t.left:
                temp = _get_codes(t.left, n) 
                if temp != '':
                    address += '0' + temp
            
            if t.right:
                temp = _get_codes(t.right, n) 
                if temp != '':
                    address += '1' + temp             
                
        return address
    
    output = {}
    
    if tree:
        
        if tree.left:
            
            if tree.left.symbol:
                output[tree.left.symbol] = _get_codes(tree, tree.left.symbol)

            else:
                temp = get_codes(tree.left)
                                
                for t in temp:
                    temp[t] = '0' + temp[t]
                    
                output.update(temp)
            
        if tree.right:
        
            if tree.right.symbol:
                output[tree.right.symbol] = _get_codes(tree, tree.right.symbol)
                
            else:
                temp = get_codes(tree.right)
                
                for t in temp:
                    temp[t] = '1' + temp[t]
                    
                output.update(temp) 
    return output

#test helper function
def _get_codes(t, n):
    ''' Find n in t and return it's binary address, or empty string if it is
    not found
    '''
    address = ''
    
    if t:

        if t.left and t.left.symbol == n:
            return '0'
        
        elif t.right and t.right.symbol == n:
            return '1'
        
        if t.left:
            temp = _get_codes(t.left, n) 
            if temp != '':
                address += '0' + temp
        
        if t.right:
            temp = _get_codes(t.right, n) 
            if temp != '':
                address += '1' + temp             
            
    return address


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
    
    #Do a postorder traversal of the tree.
    #Assign the first internal node (node with children) the number 0.
    #Every internal node found after is going to be assigned a number starting
    #from 0.
    #Reassign each 'None' or Internal Node, according to its postorder traversal
    #position number.
    
    def _internal_post(tree):
        
        #if we hit a leaf node
        
        if not tree or (not tree.left and not tree.right):
            return []
        
        return _internal_post(tree.left) + _internal_post(tree.right) + [tree]
            
    internals = _internal_post(tree)
    
    for i in range(len(internals)):
        
        internals[i].number = i

def _internal_post(tree):
    
    #if we hit a leaf node
    if not tree or (not tree.left and not tree.right):
        return []
    return _internal_post(tree.left) + _internal_post(tree.right) + [tree]

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
    divisor = sum(freq_dict.values())
    
    code_dict = get_codes(tree)
    
    for k in code_dict:
        
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

    compressed = ''
    
    for t in text:
        
        compressed += d[int(str(t))]

    output = []
    
    while len(compressed) >= 8:
        
        output.append(bits_to_byte(compressed[:8]))
        compressed = compressed[8:]
        
    if compressed != '':
        output.append(bits_to_byte('{:0<8s}'.format(compressed)))
        
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
    
    #Using postorder traversal, find the first None node
    
    def _tree_to_bytes(tree): #Return a postorder list of None nodes in tree
        
        if not tree or (not tree.left and not tree.right):
            return []
        
        return _tree_to_bytes(tree.left) + _tree_to_bytes(tree.right) + [tree]
    
    ordered = _tree_to_bytes(tree)
    
    for o in ordered:
        print(o)
        
        #If our left is a leaf node
        if o.left and (not o.left.left and not o.left.right):
            output.append(0)
            
            output.append(o.left.symbol)
            
        elif o.left:
            output.append(1)
            
            output.append(o.left.number)
            
        if o.right and (not o.right.left and not o.right.right):
            
            output.append(0)
            output.append(o.right.symbol)
            
        elif o.right:
            output.append(1)
            output.append(o.right.number)
            
    return output
    
def _tree_to_bytes(tree): #Return a postorder list of None nodes in tree
     
    if not tree or (not tree.left and not tree.right):
        return []
     
    return _internal_post(tree.left) + _internal_post(tree.right) + [tree]
    
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

def _tester_GNG(node_list, root_index):
    
    nl = node_list[:]
    
    tree_lst = []
    leaves = []
    
    root = nl.pop(root_index)
    
    for rn in nl:
        
        if rn.l_type == 0 and rn.r_type == 0:
            leaves.append(rn)

    for x in range(len(leaves)):        
        nl.remove(leaves[x])
        
    internals = []
    
    #Now we only have internal nodes left
    for rn in nl:
        t = HuffmanNode()
        if rn.l_type == 0:
            
            t.left = HuffmanNode(rn.l_data)
            
        elif rn.l_type:
            t.left = HuffmanNode()
            t.left.left = HuffmanNode(leaves[0].l_data)
            t.left.right = HuffmanNode(leaves[0].r_data)
            leaves.pop(0)
            
        if rn.r_type == 0:
            
            t.right = HuffmanNode(rn.r_data)
            
        elif rn.r_type:
            t.right = HuffmanNode()
            t.right.left = HuffmanNode(leaves[0].l_data)
            t.right.right = HuffmanNode(leaves[0].r_data)                
            leaves.pop(0)
            
        internals.append(t)
        
    #We should now have no items left in tree_lst, as they were all randomly
    #Assigned as children to the internals
    return HuffmanNode(None, internals[0], internals[1])


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
    # todo


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    # todo


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
            g.write(generate_uncompressed(tree, text, size))


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
    # todo

if __name__ == "__main__":
    
    rn_lst = [ReadNode(None,None,1,3), ReadNode(0,3,1,5), ReadNode(0,4,0,5), ReadNode(1,2,1,1), ReadNode(0,1,0,2)]
    a = _tester_GNG(rn_lst, 3)
    
    
    
    #fd = {'a':2, 'b':3, 'c':4}
    #hn = huffman_tree(fd)
    #print(hn.symbol == None and hn.number == 9 and hn.left.symbol == 'c' and 
          #hn.left.number == 4 and hn.right.symbol == None and hn.right.number ==
          #5 and hn.right.left.symbol == 'a' and hn.right.left.number == 2 and
          #hn.right.right.symbol == 'b' and hn.right.right.number == 3)
    
    #d = {0: "0", 1: "10", 2: "11"}
    #text = bytes([1, 2, 1, 0])   
    #result = generate_compressed(text, d)    
    
    #left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    #right = HuffmanNode(5)
    #tree = HuffmanNode(None, left, right)
    #number_nodes(tree)
    #print(tree_to_bytes(tree))
                        
    #tree = HuffmanNode(None, HuffmanNode(t3), HuffmanNode(2))
    #number_nodes(tree)
    #print(tree_to_bytes(tree))    
    
    #tree1 = HuffmanNode(None)
    #tree1.left = HuffmanNode(None)
    #tree1.left.left = HuffmanNode(3)
    #tree1.left.right = HuffmanNode(None)
    #tree1.left.right.left = HuffmanNode(1)
    #tree1.left.right.right = HuffmanNode(2)
    #tree1.right = HuffmanNode(None)
    #tree1.right.right = HuffmanNode(None)
    #tree1.right.right.left = HuffmanNode(4)
    #tree1.right.right.right = HuffmanNode(5)
    #number_nodes(tree1)
    #print(tree_to_bytes(tree1))
    
    
    
    #fd2 = {'a':2}
    #hn2 = huffman_tree(fd2)
    
    
    #b = HuffmanNode()
    #b.left = HuffmanNode('c')
    #b.right = HuffmanNode()
    #b.right.left = HuffmanNode()
    #b.right.left.left = HuffmanNode('a')
    #print( _get_codes(b, 'a') == '100')
    
    
    
    
    
    
    
    
    #import python_ta
    #python_ta.check_all(config="huffman_pyta.txt")
    # TODO: Uncomment these when you have implemented all the functions
    # import doctest
    # doctest.testmod()

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
        #print("uncompressed {} in {} seconds."
              #.format(fname, time.time() - start))
    
    