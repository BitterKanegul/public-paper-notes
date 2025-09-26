# Column-Oriented Datalog on the GPU
---
Here are my notes for https://sidharthkumar.io/publications/AAAI2025.pdf

## 1. Premise
  - First column oriented Datalog engine that exploits its effectiveness on the GPU
  - 200x perf gain over SOTA CPU engines
  - 2.5x per gain over GPU
  - Work loads include the KRR
## 2. Preliminaries
  - Datalog
  - Decomposed Storage Model (DSM)
    -   Store as ([(Column_i, PID)], [(Column_i+1, PID)]...) (structure of arrays)
    -   PID is the surrogate column
    -   all col is easy, a row, we join based on PID
    -   additional id increases memory, perform column compression
    -   write operation is more expensive.
    -   datalog the materialized IDB is OOM larger than the EDB
    -   merging delta with full in semi-naive evaluation needs heavy write operation
    -   mitigation in Vlog is * ON-demand Concatenation*
    -   FVlog eats the cost of data insertion to maintain better properties.
    -   
  - Column Oriented Relations on the GPU
    - Uncompressed Raw Data
    - Schedule multiple rules per iteration
  - Relational algebra operators
    - Datalog query -> extended positive RA operations
      - Join, Select, Project, Set Union + Fixpoint Closure
      - Projection and Selection
        - The projection operator processes surrogate columns instead of whole rows
        - Selection operates directly on raw data array of the targeted column
        - Implemented using the CCCL library
      - Join
        - We don't materialize the full result, only return matched surrogate columns
        - Given R_A and R_B :
        - 2 phases
          - First phase is computing join size
            -   Each gpu thread goes over data column of decomposed Reach_y
            -   sorted index helps get the range of ids that have c = a value quickly
            -   for each column c in R_A x, x = Id(c)
            -   range[x] = R_B.hashmap[c] or null
            -   matched_A[x] = x or null
            -   filter out nulls
            -   size = sum of all range.size()
            -   for and sum are in parallel
          -  R_C = parallel_allocate (size)
          -  this helps with async allocations as size compute can be done in parallel with rest of the join
          -  2 ways to collect join result
          -  GPU prefer to have the same number of tuples written at each step
            - pos_buf = exclusive_scan (size of each range)
            - for n from 0 to R_C-1
              - ub = pos_buf[j+ 1| total_size]
              - find j where pos_buff[j]<n<pb[j+1]
              - A<- matched[j]
              - B <- R_B.sorted[range[j].start + (n-pos[j])
              - write (A_id, B_id) to R_C[n]
              - can we make this multi-step into a single pipelined kernel for SoL
            - this is the result offset buffer
            - iterate over all positions in result memory in parallel
            - in each thread sequential/binary search identifies matched position in ranges
            - join result is written to the output buffer
            -  ?? Better multi-way joins / free joins here?
      -  Set Union and Deduplication
          - Complicated in DSM since dedup needs simultaneous access to entire rows
          - Extend RA(+) with difference , removes tuples from left that match right
          - We can make deltas easily
          - Triangle join : AGM bound for binary joins
          - Solution is to use leapfrog trie join classically
          - for GPUs maybe col-oriented free join
          - here instead of free joins, tailor to GPU friendly datastructure
          - hash index of the two relations
          - Deduplication in FVlog for 2-arity relation
            - New(x,y) , S(id,x), T(id,y)
            - Q is bitmap for matched new
              - for a in New_x
                - range_x <- S.hashmap[a]
              - match_id = range(R.size)
              - do same for New_y and T generate range_y
              - remove i if either empty
              - if range_x[i] and range_y[i] overlap, Q[i]=true                      

- Evaluation
  -  VLog and Nemo were the CPU- column oriented Datalog engines
  -  AMD EPYC 9534, H100
  -  Same Generation Query
  -  GPUJoin compresses rows into single 32-bit ints
  -  per column of DSM allows parallel radix sort
  -    
I wonder how they implemented a true provenance semiring...



#### Rapids Memory Manager
From: https://github.com/rapidsai/rmm
- Overhead of cudaMalloc and synchronization of cudaFree was holding RAPIDS back.
- Wraps around CNMem
- RMM is 1000 times faster than cudaMalloc and cudaFree
- customizes device and host memory allocation
- interface implementations and data structures that use the interface for memory allocation
- Primary interface to memory allocation is through memory resources, MR-> allocate, deallocate
- device MRs implement stream-ordered memory allocation.
- CUDA stream programming model...
- allocation and deallocation of memory is similar to kernel launch and stuff.
- using allocate on same memory different streams can cause weirdness unless synched
- https://leimao.github.io/blog/CUDA-Stream/
- Binning allocator optimizes for smol rapid allocations
- arena, cuda, binning, pool
- pool is less than a microsecond
- arena suffers less from fragmentation
- binning is fixed size so almost constant time
- ? a decently tight upper bound on how much memory is needed given datalog program
- built in thrust support
```
// Allocates at least 100 bytes on stream `s` using the *default*
// resource and *default* stream
rmm::device_vector<int> v{100}; 
int* p = v.data(); // typed pointer to device memory

// Data in `v` is safe to access on default stream
kernel_2<<<..., rmm::cuda_stream_default()>>>(p); 

```   
#### HISA: Hash Indexed Sorted Array
- Main workhorse of the FVlog relation joining
- Memory optimized, useful for range queries.
- k-ary joins was being developed in the GDLog paper https://arxiv.org/pdf/2311.02206v3
- They provide an example of Souffle's semi-naive evaluation (you only work on tuples that were introduced in the previous iteration)
  - e.g. Reach relation split into 3 versions: new for tuples current iteration , delta for previous iteration, full for all tuples
- Fast indexing, Range Queries and Sorting are needed for deduction heavy workloads
  - HISA is optimized for SIMD
  - Inspired by HashGraph (https://arxiv.org/pdf/1907.02900)
  - Key Steps:
    - Map between compressed data array on memory and sorted index map
    - Use a grid-stride loop in CUDA
    - First step: Build Sorted Index Map
      - 35,11,46, 97
      - join key determines order
      - sorded index map creates an open-addressing hash table with low order and inserts keys in lexi-order
      - collisions are handled by linear probing. better for the GPU
      - hashing -> murmur3
      - cuda atomic operation for parallel hash table construction
      - an example is described in: https://developer.nvidia.com/blog/maximizing-performance-with-massively-parallel-hash-maps-on-gpus/
      - index map has them sorted, (*11, *35, *46, *97)
      - range query, deduplication is simplified.
      - Hashgraph demos the effectiveness of combining a hashmap with compact array-style datastructure
      - But for Datalog we need deduplication and handle multi-values
      - HISA helps by making a sorted hash indexed array that maps to the compact representation
- Inductive queries in DL
  - Fast range queries are needed
  - HISA helps with the serialization requirements of recursive joins
  - Join column data only helps with efficient insertion and retrieval ops
  - Serialization can be done by parallel iteration over this array
  - by keeping only the columns that matter for the join memory is saved. (FVLog does this naturally as it is a columnar store)
  - by doing essentially "pointer arithmetic" one can reduce the number of times we need to load things from main memory
- HISA Impl:
```
using column_type=rmm::device_vector<u32>;
struct GHashRelContainer {
// open addressing hashmap for indexing
OpenAddressMap *index_map = nullptr;
tuple_size_t index_map_size = 0;
float index_map_load_factor;
// flatten tuple data
device_vector<tuple_size_t> sorted_scalar;
// will be inited to arity size
column_type *data_raw = nullptr;
tuple_size_t tuple_counts = 0;
int arity; bool tmp_flag = false;
};
__global__ void load_relation_container(
GHashRelContainer *target,int arity,column_type &data,
tuple_size_t data_size,tuple_size_t index_column_size,
float index_map_load_factor);
__global__ void create_index(
GHashRelContainer *target, tuple_indexed_less cmp);
__global__ void get_join_result_size(
OpenAddressMap *inner, column_type &outer,
int join_column_counts,tuple_size_t *join_result_size);
__global__ void get_join_result(
OpenAddressMap *inner, column_type &outer, int arity,
int join_column_counts, column_type *output_raw_data);
```
  - data_raw has the compact array for sorted data
  - load_relation_container does GPU sorting and stores the tuples in vram
  - create_idx is a kernel that creates an index based on loaded raw data within HISA object
  - HISA being built for deductive reasoning offers
    - get_join_result and get_join_result_size
    - take inner relation hashtable and outer relation sorted array as input
    - coalesced memory access speeds it up and improves throughput
    - join_result uses bulk data retrieval
    - precalculating join sizes help since join result sizes vary through the operation
    - (maybe a tight upper_bound / lower_bound would help?)
    
      
### Free Join: Unifying Worst-Case Optimal and Traditional Joins


Generic Join on the gpu:

How do you represent the join as a matmul?

- Subtract to get 0 is the way/ do the graph adjacency matrix thing.


https://arxiv.org/pdf/2301.10841
 WCOJ miscategorized as working for cyclic graphs, and yannalakis for the acyclic version.
 
 
 These guys unify the WCOJ with the traditional Binary Join.
 They create a new datastructure called Column Oriented Lazy Trie.

Classic Column oriented Layout to improve the trie data structure in WCOJ.

Propose a vectorized execution algorithm for the Free Join.
- Left-deep binary join is similar to the Generic Join.

Two algorithms process the join operation similarly:
   -  Binary Hash Join iterates over tuples on one relation.
        - For each tuple, probes into the hash table of other relation.
   - Each loop level in Generic Join iterates over the keys of a certain trie.
   - Probes into several other tries for each key.
    
Free join takes an optimized binary join,
    converts to the free join plan,
     optimizes the free join plan.
       something that sits between the free join and generic join
   
   - takes full advantage of design space (not restricted to joining one tuple at a time)
   - uses existing cost based optimizers for binary joins.
   
   - Main inefficiency of generic join: constructing trie on each relation of the query.
   - Binary join map only needs to build hashmap for the right hand side relation of a join.
   - Need to improve trie building speed.
   - One optimization is they don't build tries for tables that are left children.
   
   - COLT datastructure builds tries lazily, building subtries on demand.
- Much dual, very nice
```
a left-deep linear plan for binary join is a
sequence of relations; it need not specify the join attributes, since
all shared attributes are joined. In contrast, a Generic Join plan is
a sequence of variables; it need not specify the relations, since all
relations on each variable are joined.
```

Ok, so they partition the query Q's variables into different subsets, and order these partitions.
Every $\phi_k$ has access to the variables before it $\phi_i <_{Q} \phi_k$.


Example free plan is
$[[R(x,a)], [S(b), T(x)], [T(c)]]$

to execute first node, we iterate over each tuple (x,a) in R
use x to probe into S
iterate over each b in S[x]
use x to probe into T,
finally iterate over all c in T[x]

This is the left-deep plan

For the generic join plan we can do
$[[R(x),S(x), T(x) ], [R(a)], [S(b)], [T(c)]]$

Now I see how this can move between the two different plans and how one can trade off...

Then the algorithm is executed: 

```
 fn join(all_tries, plan, tuple):
 if plan == []:
 output(tuple)
 else:
 tries = [ t ∈ all_tries | t.relation ∈ plan[0] ]
 # iterate over the cover
 @outer for t in tries[0].iter():
 subtries = [ iter_r.get(t) ]
 tup = tuple + t
 # probe into other tries
 for trie in tries[1..]:
 key = tup[trie.vars]
 subtrie = trie.get(key)
 if subtrie == None: continue @outer
 subtries.push(subtrie)
 new_tries = all_tries[tries  → subtries]
 join(new_tries, plan[1:], tup)
```

Optimizing Free Join Plan:





