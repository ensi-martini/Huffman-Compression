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
        
        node_list = []
        
        #Add every key and value in the freq dict as a HNode with their proper
        #number values
        for n in freq_dict:

            #Letters that do not appear but are in the dict for some stupid
            #reason do not get added
            if freq_dict[n] > 0:
                node = HuffmanNode(n)
                node.number = freq_dict[n]
                node_list.append(node)
                
        node_list.sort(key=lambda x: x.number)
        
        #After sorting from smallest to greatest occurances, iterate each time
        #and merge the two least frequent nodes together (could be internals)
        while len(node_list) > 1:
            
            #Just for testing: REMOVE THIS LATER
            #name = node_list[0].symbol + node_list[1].symbol
                                       
            current_node = HuffmanNode(None, node_list[0], node_list[1])
            current_node.number = node_list[0].number + node_list[1].number
            
            node_list = node_list[2:] + [current_node]
            node_list.sort(key=lambda x: x.number)
            
        return node_list[0]
        
    root = HuffmanNode(None)
    
    if len(freq_dict) == 1:
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

    lines = []    
    cache = {}

    for t in text:
        lines.append(codes[t])
        
    compressed = ''.join(lines)
    length = len(compressed)
    remainder = ( length % 8)  * -1
    remaining = compressed[remainder:]
    compressed = compressed[:remainder]
    #IMPROVEMENTS:
    #Check last byte using lambda
    
    output = []
    
    for i in range(0, length + remaining, 8):
        current = compressed[i:i + 8]
        
        if current not in cache:
            cache[current] = bits_to_byte(current)
        output.append(cache[current])
    
    print('done for loop!')

    if remaining != '':
        
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
    
    #Using postorder traversal, find the first None node
    
    def _tree_to_bytes(tree): #Return a postorder list of None nodes in tree
        
        if not tree or (not tree.left and not tree.right):
            return []
        
        return _tree_to_bytes(tree.left) + _tree_to_bytes(tree.right) + [tree]
    
    ordered = _tree_to_bytes(tree)
    
    for o in ordered:
        #print(o)
        
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
            
    return bytes(output)
    
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
    print('done freq!')
    tree = huffman_tree(freq)
    print('done huff_tree!')
    codes = get_codes(tree)
    print('done codes!')
    number_nodes(tree)
    print('done num_nodes!')
    print("Bits per symbol:", avg_length(tree, freq))
    #result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
     #         size_to_bytes(len(text)))
    r1 = num_nodes_to_bytes(tree)
    print('done num_nodes to bytes!')
    r2 = tree_to_bytes(tree)
    print('done tree to bytes!')
    r3 = size_to_bytes(len(text))
    print('done size to bytes!')
    result = (r1 + r2 + r3)
    result += generate_compressed(text, codes)
    print('done generate_compressed')
    with open(out_file, "wb") as f2:
        f2.write(result)

    print('done!')

# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12)]
    >>> lst.append(ReadNode(1, 1, 1, 0))
    >>> a = generate_tree_general(lst, 2)
    >>> b = HuffmanNode()
    >>> b.right = HuffmanNode(None, HuffmanNode(10), HuffmanNode(12))
    >>> b.left = HuffmanNode(None, HuffmanNode(5), HuffmanNode(7))
    >>> a == b
    True
    """
    
    nodes = node_lst[:]
    
    root = nodes.pop(root_index)
    
    #We know that each node has it's node number at the same place as it's
    #index, so if our root is not the last item in the list we need to
    #put some placeholder to compensate
    
    nodes.insert(root_index, None)
    
    root_node = HuffmanNode()
    
    if root.l_type == 1:
        root_node.left = generate_tree_general(nodes, root.l_data)
        
    else:
        root_node.left = HuffmanNode(root.l_data)
        
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
    
    nodes = node_lst[:]
    
    root = nodes[root_index]
    
    #We know that each index is the postorder equivalent of the parent at that 
    #subtree
    
    #We also know that the root's right child, if it is a subtree, has a root 
    #with a postorder value of root_index - 1
    
    #The root node will always be the last one, in postorder
    
    #We cannot use the l_data or r_data properties for internals
    
    root_node = HuffmanNode('{}:{}'.format('N',root_index))
    
    if root.r_type == 1:
        
        root_node.right = generate_tree_postorder(nodes[:-1], root_index - 1)
        
    else:
        root_node.right = HuffmanNode(root.r_data)
        
    #After we have reached the end of the right side, the 
    #next found one on the left will be the one before in the postorder repr
    
    if root.l_type == 1:
        
        #Ok so here is where i get stuck: we know that we can keep decreasing
        #the postorder number, problem is that we dont know how many were
        #already added on the right subtree so we dont know where to splice
        #the list. We could make a helper function that counts the number of 
        #internals and subtracts that
        
        #print(root_node)
        internals = count_internals(root_node)
        root_node.left = generate_tree_postorder(nodes[:-1 * internals], \
                                                 root_index - 1 - internals)
        
    else:
        root_node.left = HuffmanNode(root.l_data)
        
    return root_node
        
def count_internals(t):
    
    total = 0
    
    if t:
        
        if t.left and t.left.left and t.left.right:
            total += count_internals(t.left) + 1
            
        if t.right and t.right.left and t.right.right:
            total += count_internals(t.right) + 1
            
    return total
            
def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompressd
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    
    #We take a tree, compressed bytes like the output in our generate_compressed
    #and a number of items to decode, to avoid adding padding to output
    
    
    codes = get_codes(tree)
    print('done get_codes!')
    inverse_codes = {value: key for key, value in codes.items()}

    bit_form = []
    cache = {}
    
    for b in text:
        if b not in cache:
            cache[b] = byte_to_bits(b)
        bit_form.append(cache[b])
        
    bit_form = ''.join(bit_form)
    
    print('done first for loop!')
    output = []
    
    for char in range(size):
        
        key = bit_form[0]
        counter = 0
        
        while key not in inverse_codes:
            counter += 1
            key += bit_form[counter]
            
        bit_form = bit_form[counter + 1:]
            
        output.append(inverse_codes[key])
        
    print('done second loop!')
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
        print('done bytes_to_nodes!')
        #print(node_lst)
        print()
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        print('done generate_tree_general!')
        #print(tree)
        print()
        size = bytes_to_size(f.read(4))
        print('done bytes_to_size!')
        #print(size)
        print()
        with open(out_file, "wb") as g:
            text = f.read()
            #print(tree)
            #print(text)
            #print(size)
            something = generate_uncompressed(tree, text, size)
            
            print(something == None)
            g.write(something)
            
        print('done!')


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

#if __name__ == "__main__":
    #cProfile.run('compress("b.txt", "b.txt.huf")')
    #cProfile.run('uncompress("b.txt.huf", "out.txt")')
    
    
    
    #import python_ta
    #python_ta.check_all(config="huffman_pyta.txt")
    # TODO: Uncomment these when you have implemented all the functions
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
        #print("uncompressed {} in {} seconds."
              #.format(fname, time.time() - start))
    
    