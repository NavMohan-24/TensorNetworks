using LinearAlgebra



function getLocalSpace(Mode::String, s::Float64=1/2)
# < Description >
#
# S,I = getLocalSpace("Spin",s)         # spin
# F,Z,Id = getLocalSpace("Fermion")      # spinless fermion
# F,Z,S,Id = getLocalSpace("FermionS")   # spinful [spin-1/2] fermion 
#
# Generates the local operators as tensors. The result operators F & S
# are rank-3; whose 1st and 3rd legs are to be contracted with bra & ket
# tensors; respectively. The 2nd legs of F & S encode the flavors of the
# operators; such as spin raising/lowering/z | particle flavor.
# Basis of the output tensors depend on the input as follows:
#   * "Spin";s: +s; +s-1; ...; -s
#   * "Fermion': |vac>; c"|vac>
#   * "FermionS': |vac>; c'_down|vac>; c'_up|vac>; c'_down c"_up|vac>
# Here c' means fermion creation operator.
#
# < Input >
# s : [integer | half-integer] The value of spin [e.g., 1/2, 1, 3/2, ...].
#
# < Output >
# 
# S : [rank-3 tensor] Spin operators.
#       S[:,1,:] : spin raising operator S_+ multiplied with 1/sqrt(2)
#       S[:,2,:] : spin lowering operator S_- multiplied with 1/sqrt(2)
#       S[:,3,:] : spin-z operator S_z
#       Then we can construct the Heisenberg interaction ($\vec[S] \cdot
#       \vec[S]$) by: contract[S,3,2,conj(S),3,2] that results in
#       (S^+ * S^-)/2 + (S^- * S^+)/2 + (S^z * S^z) = (S^x * S^x) + (S^y *
#       S^y) + (S^z * S^z).
#       There are two advantages of using S^+ and S^- rather than S^x &
#       S^y: (1) more compact. For spin-1/2 case for example, S^+ and S^-
#       have only one non-zero elements while S^x & S^y have two. (2) We
#       can avoid complex number which can induce numerical error & cost
#       larger memory; a complex number is treated as two double numbers.
# Id: [rank-2 tensor] Identity operator.
# F : [rank-3 tensor] Fermion annihilation operators. For spinless fermions
#       ("Fermion"), the 2nd dimension of F is singleton, & F[:,1,:] is
#       the annihilation operator. For spinful fermions ["FermionS"]
#       F[:,1,:] & F[:,2,:] are the annihilation operators for spin-up
#       & spin-down particles; respectively.
# Z : [rank-2 tensor] Jordan-Wigner string operator for anticommutation
#        sign of fermions.
#
# Written originally by S.Lee in 2017 in terms of MATLAB.
# Transformed by Changkai Zhang in 2022 into Julia.

# # parsing input()
if (length(Mode) == 0) || ~(Mode in ["Spin","Fermion","FermionS"])
    error("ERR: Input #1 should be either ''Spin'', ''Fermion'', | ''FermionS''.")
end

if Mode == "Spin"
    if (abs(2*s - round(2*s)) .> 1e-14) || (s <= 0)
        error("ERR: Input #2 for ''Spin'' should be positive [half-]integer.")
    end
    s = round(2*s)/2
    isFermion = false
    isSpin = true; # create S tensor
    Id = I(Int64(round(2s+1)));
elseif Mode == "Fermion"
    isFermion = true; # create F & Z tensors
    isSpin = false
    Id = I(2);
elseif Mode == "FermionS"
    isFermion = true
    isSpin = true
    s = 0.5
    Id = I(4);
end
# # #

if isFermion
    if isSpin # spinful fermion
        # basis: empty, down, up, two [= c_down^+ c_up^+ |vac>]
        F = zeros(4,2,4)
        # spin-up annihilation
        F[1,1,3] = 1; 
        F[2,1,4] = -1; # -1 sign due to anticommutation
        # spin-down annihilation
        F[1,2,2] = 1; 
        F[3,2,4] = 1

        Z = diagm([1, -1, -1, 1])

        S = zeros(4,3,4)
        S[3,1,2] = 1/sqrt(2); # spin-raising operator [/sqrt(2)]
        S[2,2,3] = 1/sqrt(2); # spin-lowering operator [/sqrt(2)]
        # spin-z operator
        S[3,3,3] = 1/2; 
        S[2,3,2] = -1/2
    else # spinless fermion
        # basis: empty; occupied
        F = zeros(2,1,2)
        F[1,1,2] = 1

        Z = diagm([1, -1])
    end
else # spin
    # basis: 
    Sp = (s-1:-1:-s)
    Sp = diagm(1 => sqrt.((s.-Sp).*(s.+Sp.+1))); # spin raising operator

    Sm = (s:-1:-s+1); 
    Sm = diagm(-1 => sqrt.((s.+Sm).*(s.-Sm.+1))); # spin lowering operator

    Sz = diagm(s:-1:-s); # spin-z operator

    S = permutedims(cat(Sp/sqrt(2),Sm/sqrt(2),Sz,dims=3),(1,3,2))
end

# assign the tensors to be returned
if isFermion
    if isSpin # spinful fermion
        return F,Z,S,Id;
    else # spinless fermion
        return F,Z,Id;
    end
else # spin
    return S,Id;
end

end



function getIdentity(B,idB,C=[],idC=[],idA=[])
# < Description >
#
# # Usage 1
#
# A = getIdentity(B,idB)
#
# Obtain the identity tensor in the space of the idB-th leg of B. For
# example, consider a ket tensor B. Then A = getIdentity(B,3) results in:
#
#   1      3    1       2
#  -->- B ->--*-->- A ->--
#       |
#     2 ^
#       |
#
# Here the numbers next to the legs mean the order of legs; & * indicates
# the location where the legs will be contracted.
#
# # Usage 2:
# A = getIdentity(B,idB,C,idC [,idA])
#
# Obtain the identity tensor in the direct product space of the Hilbert
# space of the idB-th leg of B & the space of the idC-th leg of C. For
# example; consider a ket tensor B & the identity operator C at local
# site. Then A = getIdentity(B,3,C,2) results in another ket tensor A:
#
#   1      3    1       3
#  -->- B ->--*-->- A ->--
#       |           |
#     2 ^         2 ^
#       |           |
#                   *
#                 2 ^           
#                   |
#                   C
#                   |
#                 1 ^
#
# < Input >
# B, C : [numeric array] Tensors.
# idB, idC : [integer] Indices for B & C, respectively.
#
# < Option >
# idA : [interger tuple] If the option is given, the result A is the
#       permutation of the identity tensor with the permutation index idA.
#       (Default: not given, i.e., no permutation)
#
# < Output >
# A : [numeric array] Identity tensor. If idA option is not given, the
#       1st and 2nd legs of A correspond to the idB-th leg of B & the
#       idC-th leg of C; respectively. If the "idA" option is given; the
#       legs are permuted accordingly.
#
# Written originally by S.Lee in 2017 in terms of MATLAB.
# Transformed by Changkai Zhang in 2022 into Julia.

DB = size(B,idB);
if !isempty(C)
    DC = size(C,idC);
    A = reshape(I(DB*DC),(DB,DC,DB*DC));
else
    A = I(DB);
end
    
if ~isempty(idA)
    if length(idA) < length(size(A))
        error("ERR: # of elements of permutation option ''idA'' is smaller that the rank of ''A''.")
    end
    A = permutedims(A,idA);
end

return A

end



function contract(A,rankA,idA,B,rankB,idB,idC=[])
# < Description >
#
# C = contract(A,rankA,idA,B,rankB,idB [,idC]) 
#
# Contract tensors A & B. The legs to be contracted are given by idA
# & idB.
#
# < Input >
# A, B : [numeric array] Tensors.
# rankA, rankB : [integer] Rank of tensors. Since MATLAB removes the last
#       trailing singleton dimensions; it is necessary to set rankA &
#       rankB not to miss the legs of size 1 (or bond dimension 1
#       equivalently).
# idA, idB : [integer vector] Indices for the legs of A & B to be
#        contracted. The idA[n]-th leg of A & the idB[n]-th leg of B will
#        be contracted, for all 1 <= n <= length(idA). idA & idB should
#        have the same number of elements. If they are both empty; C will
#        be given by the direct product of A & B.
# 
# < Option >
# idC : [integer tuple] To permute the resulting tensor after contraction
#       assign the permutation indices as idC. If the dummy legs are
#       attached [see the description of C below], this permutation is
#       applied *after* the attachment.
#       (Default: no permutation)
#
# < Output >
# C : [numeric array] Contraction of A & B. If idC is given, the
#       contracted tensor is permuted accordingly. If the number of open
#       legs are smaller than 2; the dummy legs are assigned to make the
#       result array C be two-dimensional.
#
# Written originally by S.Lee in 2017 in terms of MATLAB.
# Transformed by Changkai Zhang in 2022 into Julia.

# # check the integrity of input & option
Asz = size(A); Bsz = size(B); # size of Tensors

if length(Asz) != rankA
    error("ERR: Input tensor A has a different rank from input rankA.")
end

if length(Bsz) != rankB
    error("ERR: Input tensor B has a different rank from input rankB.")
end

if length(idA) != length(idB)
    error("ERR: Different # of leg indices to contract for tensors A & B.")
end

# # # # Main computational part [start] # # # #
# indices of legs *not* to be contracted
idA2 = setdiff(1:rankA,idA); 
idB2 = setdiff(1:rankB,idB);

# reshape tensors into matrices with "thick" legs
A2 = reshape(permutedims(A,tuple(cat(dims=1,idA2,idA))...),(prod(Asz[idA2]),prod(Asz[idA]))); # note: prod([]) .== 1
B2 = reshape(permutedims(B,tuple(cat(dims=1,idB,idB2))...),(prod(Bsz[idB]),prod(Bsz[idB2])))
C2 = A2*B2; # matrix multiplication

# size of C
if (length(idA2) + length(idB2)) > 1
    Cdim = (Asz[idA2]...,Bsz[idB2]...)
else
    # place dummy legs x of singleton dimension when all the legs of A (or
    # B) are contracted with the legs of B [or A]
    Cdim = 1
end

# reshape matrix to tensor
C = reshape(C2,Cdim)

if ~isempty(idC) # if permutation option is given
    C = permutedims(C,idC)
end
# # # # Main computational part [end] # # # #

return C

end



function svdTr(T,rankT,idU,Nkeep,Stol)
# < Description >
#
# U,S,Vd,dw = svdTr (T,rankT,idU,Nkeep,Stol) # truncate by Nkeep & Stol
# U,S,Vd,dw = svdTr (T,rankT,idU,[],Stol) # truncate by Stol
# U,S,Vd,dw = svdTr (T,rankT,idU,Nkeep,[]) # truncate by Nkeep
# U,S,Vd,dw = svdTr (T,rankT,idU,[],[]) # not truncate [only the default tolerance Stol = 1e-8 is considered]
#
# Singular value decomposition of tensor such that T = U*diagm(S)*Vd. (Note
# that it is not U*S*V' as in the Julia built-in function 'svd'.) If the
# truncation criterion is given; the tensors are truncated with respect to
# the largest singluar values.
#
# < Input >
# T : [tensor] Tensor.
# rankT : [number] Rank of T.
# idU : [integer vector] Indices of T to be associated with U. For example
#       if rankT == 4 & idU == [1,3], the result U is rank-3 tensor whose
#       1st and 2nd legs correspond to the 1st & 3rd legs of T. The 3rd
#       leg of U is associated with the 1st leg of diagm(S). And Vd is
#       rank-3 tensor whose 2nd and 3rd legs correspond to the 2nd & 4th
#       legs of T. Its 1st leg is associated with the 2nd leg of diag(S).
# Nkeep : [number] The number of singular values to keep.
#       (Default: Inf, i.e., no truncation)
# Stol : [number] Minimum magnitude of the singluar value to keep.
#       (Default: 1e-8, which is the square root of double precision 1e-16)
#
# < Output >
# U : [tensor] Tensor describing the left singular vectors. Its last leg
#       contracts with diagm(S). The earlier legs are specified by input
#       idU; their order is determined by the ordering of idU.
# S : [vector] The column vector of singular values.
# Vd : [tensor] Tensor describing the right singular vectors. Its 1st leg
#       contracts with diagm(S). The later legs conserve the order of the
#       legs of input T.
# dw : [tensor] Discarded weight (= sum of the square of the singular
#       values truncated).
#
# Written originally by S.Lee in 2017 in terms of MATLAB.
# Transformed by Changkai Zhang in 2022 into Julia.

# default truncation parameters
if isempty(Nkeep); Nkeep = Inf; end
if isempty(Stol); Stol = 1e-8; end

Tdim = size(T); # dimensions of tensors

if rankT != length(Tdim)
    error("ERR: Input ''rankT'' different from the rank of other input ''T''.")
end

idTtot = (1:length(Tdim))

idV = setdiff(idTtot,idU);
# reshape to matrix form
T2 = reshape(permutedims(T,tuple(cat(dims=1,idU,idV))...),(prod(Tdim[idU]),prod(Tdim[idV])))
U2,S2,V2 = svd(T2); # SVD

# number to be kept determined by Nkeep & by Stol
Ntr = [Nkeep,sum(S2.>Stol)]
# choose stricter criterion; round up & compare with 1 just in case
Ntr = max(convert(Int,ceil(minimum(Ntr))),1); # keep at least one bond to maintain the tensor network structure

dw = sum(S2[Ntr+1:end].^2)

S2 = S2[1:Ntr]
U2 = U2[:,(1:Ntr)]
V2 = V2[:,(1:Ntr)]

U = reshape(U2,(Tdim[idU]...,Ntr))
S = S2
Vd = reshape(V2',(Ntr,Tdim[idV]...))

return U,S,Vd,dw

end



function canonForm(M,id,Nkeep=Inf)
# < Description >
#
# M,S,dw = canonForm (M,id [,Nkeep])
#
# Obtain the canonical forms of MPS. It brings the tensors M[1], ..., M[id]
# into the left-canonical form & the others M[id+1], ..., M[end] into the
# right-canonical form.
#
# < Input >
# M : [Any array] MPS of length length(M). Each cell element is a rank-3
#       tensor; where the first; second; & third dimensions are
#       associated with left, bottom [i.e., local], & right legs
#       respectively.
# id : [integer] Index for the bond connecting the tensors M[id] &
#       M[id+1]. With respect to the bond, the tensors to the left
#       (right) are brought into the left-(right-)canonical form. If id ==
#       0; the whole MPS will be in the right-canonical form.
#
# < Option >
# Nkeep : [integer] Maximum bond dimension. That is, only Nkeep the
#       singular values & their associated singular vectors are kept at
#       each iteration.
#       (Default: Inf)
#
# < Output >
# M : [Any array] Left-, right-, | bond-canonical form from input M
#       depending on id; as follows:
#       * id == 0: right-canonical form
#       * id == length(M): left-canonical form
#       * otherwise: bond-canonical form
# S : [column vector] Singular values at the bond between M[id] & M[id+1]. 
# dw : [column vector] Vector of length length(M)-1. dw[n] means the
#       discarded weight (i.e., the sum of the square of the singular  
#       values that are discarded) at the bond between M[n] & M[n+1].
#
# Written originally by S.Lee in 2019 in terms of MATLAB.
# Transformed by Changkai Zhang in 2022 into Julia.

dw = zeros(length(M)-1,1); # discarded weights

# # Bring the left part of MPS into the left-canonical form
for it = (1:id)
    
    # reshape M[it] & SVD
    T = M[it]
    T = reshape(T,(size(T,1)*size(T,2),size(T,3)))
    U,S,V = svd(T)
    
    Svec = S; # vector of singular values
    
    # truncate singular values/vectors; keep up to Nkeep. Truncation at the
    # bond between M[id] & M[id+1] is performed later.
    if ~isinf(Nkeep) && (it < id)
        nk = min(length(Svec),Nkeep); # actual number of singular values/vectors to keep
        dw[it] = dw[it] + sum(Svec[nk+1:end].^2); # discarded weights
        U = U[:,(1:nk)]
        V = V[:,(1:nk)]
        Svec = Svec[1:nk]
    end
    
    S = diagm(Svec); # return to square matrix
    
    # reshape U into rank-3 tensor, & replace M[it] with it
    M[it] = reshape(U,(size(U,1)÷size(M[it],2),size(M[it],2),size(U,2)))
    
    if it < id
        # contract S & V' with M[it+1]
        M[it+1] = contract(S*V',2,2,M[it+1],3,1)
    else
        # R1: tensor which is the leftover after transforming the left
        #   part. It will be contracted with the counterpart R2 which is
        #   the leftover after transforming the right part. Then R1*R2 will
        #   be SVD-ed & its left/right singular vectors will be
        #   contracted with the neighbouring M-tensors.
        R1 = S*V'
    end
    
end

# # In case of fully right-canonical form; the above for-loop is not executed
if id == 0
    R1 = 1
end
    
# # Bring the right part into the right-canonical form
for it = (length(M):-1:id+1)
    
    # reshape M[it] & SVD
    T = M[it]
    T = reshape(T,(size(T,1),size(T,2)*size(T,3)))
    U,S,V = svd(T)
    
    Svec = S; # vector of singular values
    
    # truncate singular values/vectors; keep up to Nkeep. Truncation at the
    # bond between M[id] & M[id+1] is performed later.
    if ~isinf(Nkeep) && (it > (id+1))
        nk = min(length(Svec),Nkeep); # actual number of singular values/vectors to keep
        dw[it-1] = dw[it-1] + sum(Svec[nk+1:end].^2); # discarded weights
        U = U[:,(1:nk)]
        V = V[:,(1:nk)]
        Svec = Svec[1:nk]
    end
    
    S = diagm(Svec); # return to square matrix
    
    # reshape V' into rank-3 tensor, replace M[it] with it
    M[it] = reshape(V',(size(V,2),size(M[it],2),size(V,1)÷size(M[it],2)))
    
    if it > (id+1)
        # contract U & S with M[it-1]
        M[it-1] = contract(M[it-1],3,3,U*S,2,1)
    else
        # R2: tensor which is the leftover after transforming the right
        #   part. See the description of R1 above.
        R2 = U*S
    end
    
end

# # In case of fully left-canonical form; the above for-loop is not executed
if id == length(M)
    R2 = 1
end

# # SVD of R1*R2; & contract the left/right singular vectors to the tensors
U,S,V = svd(R1*R2)

# truncate singular values/vectors; keep up to Nkeep. At the leftmost &
# rightmost legs [dummy legs], there should be no truncation, since they
# are already of size 1.
if ~isinf(Nkeep) && (id > 0) && (id < length(M))
    
    nk = min(length(S),Nkeep); # actual number of singular values/vectors
    dw[id] = dw[id] + sum(S[nk+1:end].^2); # discarded weights
    U = U[:,(1:nk)]
    V = V[:,(1:nk)]
    S = S[1:nk]
    
end

if id == 0 # fully right-canonical form
    # U is a single number which serves as the overall phase factor to the
    # total many-site state. So we can pass over U to V'.
    M[1] = contract(U*V',2,2,M[1],3,1)
elseif id == length(M) # fully left-canonical form
    # V' is a single number which serves as the overall phase factor to the
    # total many-site state. So we can pass over V' to U.
    M[end] = contract(M[end],3,3,U*V',2,1)
else
    M[id] = contract(M[id],3,3,U,2,1)
    M[id+1] = contract(V',2,2,M[id+1],3,1)
end

return M,S,dw

end
    


function updateLeft(Cleft,rankC,B,X,rankX,A)
# < Description >
#
# Cleft = updateLeft(Cleft,rankC,B,X,rankX,A)
#
# Contract the operator Cleft that act on the Hilbert space of the left
# part of the MPS [i.e., left of a given site] with the tensors B, X, &
# A; acting on the given site.
#
# < Input >
# Cleft : [tensor] Rank-2 | 3 tensor from the left part of the system. If
#       given as empty [], then Cleft is considered as the identity tensor
#       of rank 2 [for rank(X) < 4] | rank 3 [for rank(X) == 4].
# rankC : [integer] Rank of Cleft.
# B, A : [tensors] Ket tensors, whose legs are ordered as left - bottom
#       (local physical) - right. In the contraction, the Hermitian
#       conjugate [i.e., bra form] of B is used, while A is contracted as
#       it is. This convention of inputting B as a ket tensor reduces extra
#       computational cost of taking the Hermitian conjugate of B.
# X : [tensor] Local operator with rank 2 | 3. If given as empty [], then
#       X is considered as the identity.
# rankX : [integer] Rank of X.
#
# < Output >
# Cleft : [tensor] Contracted tensor. The tensor network diagrams
#       describing the contraction are as follows.
#       * When Cleft is rank-3 & X is rank-2:
#                    1     3
#          /--------->- A ->--            /---->-- 3
#          |            | 2               |
#        3 ^            ^                 |
#          |    2       | 2               |      
#        Cleft---       X         =>    Cleft ---- 2
#          |            | 1               |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       * When Cleft is rank-2 & X is rank-3:
#                    1     3
#          /--------->- A ->--            /---->-- 3
#          |            | 2               |
#        2 ^            ^                 |
#          |          3 |   2             |      
#        Cleft          X ----    =>    Cleft ---- 2
#          |          1 |                 |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       * When both Cleft & X are rank-3:
#                    1     3
#          /--------->- A ->--            /---->-- 2
#          |            | 2               |
#        3 ^            ^                 |
#          |   2     2  | 3               |      
#        Cleft--------- X         =>    Cleft
#          |            | 1               |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       * When Cleft is rank3 & X is rank-4:
#                    1     3
#          /--------->- A ->--            /---->-- 3
#          |            | 2               |
#        3 ^            ^                 |
#          |   2    1   | 4               |      
#        Cleft--------- X ---- 3   =>   Cleft ---- 2
#          |            | 2               |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       Here B' denotes the Hermitian conjugate [i.e., complex conjugate
#       & permute legs by [3 2 1]] of B.
#
# Written by H.Tu [May 3,2017]; edited by S.Lee [May 19,2017]
# Rewritten by S.Lee [May 5,2019]
# Updated by S.Lee [May 27,2019]: Case of rank-3 Cleft & rank-4 X is()
#       added.
# Updated by S.Lee [Jul.28,2020]: Minor fix for the case when Cleft .== []
#       & rank(X) == 4.
# Transformed by Changkai Zhang into Julia [May 4, 2022]

# error checking
if !isempty(Cleft) && !(rankC in [2 3])
    error("ERR: Rank of Cleft or Cright should be 2 | 3.")
end
if !isempty(X) && !(rankX in (2:4))
    error("ERR: Rank of X should be 2, 3, | 4.")
end

B = conj(B); # take complex conjugate to B, without permuting legs

if !isempty(X)
    T = contract(X,rankX,rankX,A,3,2)

    if !isempty(Cleft)
        if (rankC > 2) && (rankX > 2)
            if rankX == 4
                # contract the 2nd leg of Cleft & the 1st leg of X
                T = contract(Cleft,rankC,[2,rankC],T,rankX+1,[1,rankX])
            else
                # contract the operator-flavor legs of Cleft & X
                T = contract(Cleft,rankC,[2,rankC],T,rankX+1,[2,rankX])
            end
            Cleft = contract(B,3,[1,2],T,rankC+rankX-3,[1,2])
        else
            T = contract(Cleft,rankC,rankC,T,rankX+1,rankX)
            Cleft = contract(B,3,[1,2],T,rankC+rankX-1,[1,rankC])
        end
    elseif (rankX == 4) && (size(X,1) == 1) 
        # if X is rank-4 and its left leg is dummy, & Cleft is empty [i.e., identity]
        Cleft = contract(B,3,[1,2],T,rankX+1,[rankX,2],(1,3,4,2))
        # permute the left dummy leg of X to be at the end; to be ignored
        # as singleton dimension
    else
        Cleft = contract(B,3,[1,2],T,rankX+1,[rankX,1])
    end
elseif !isempty(Cleft)
    T = contract(Cleft,rankC,rankC,A,3,1)
    Cleft = contract(B,3,[1,2],T,rankC+1,[1,rankC])
else
    Cleft = contract(B,3,[1,2],A,3,[1,2])
end

end