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

### Free Join: Unifying Worst-Case Optimal and Traditional Joins
