�
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-2-g0b15fdfcb3f8��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
v/dense_24/biasVarHandleOp*
_output_shapes
: * 

debug_namev/dense_24/bias/*
dtype0*
shape:* 
shared_namev/dense_24/bias
o
#v/dense_24/bias/Read/ReadVariableOpReadVariableOpv/dense_24/bias*
_output_shapes
:*
dtype0
�
m/dense_24/biasVarHandleOp*
_output_shapes
: * 

debug_namem/dense_24/bias/*
dtype0*
shape:* 
shared_namem/dense_24/bias
o
#m/dense_24/bias/Read/ReadVariableOpReadVariableOpm/dense_24/bias*
_output_shapes
:*
dtype0
�
v/dense_24/kernelVarHandleOp*
_output_shapes
: *"

debug_namev/dense_24/kernel/*
dtype0*
shape
:@*"
shared_namev/dense_24/kernel
w
%v/dense_24/kernel/Read/ReadVariableOpReadVariableOpv/dense_24/kernel*
_output_shapes

:@*
dtype0
�
m/dense_24/kernelVarHandleOp*
_output_shapes
: *"

debug_namem/dense_24/kernel/*
dtype0*
shape
:@*"
shared_namem/dense_24/kernel
w
%m/dense_24/kernel/Read/ReadVariableOpReadVariableOpm/dense_24/kernel*
_output_shapes

:@*
dtype0
�
v/p_re_lu_19/alphaVarHandleOp*
_output_shapes
: *#

debug_namev/p_re_lu_19/alpha/*
dtype0*
shape:@*#
shared_namev/p_re_lu_19/alpha
u
&v/p_re_lu_19/alpha/Read/ReadVariableOpReadVariableOpv/p_re_lu_19/alpha*
_output_shapes
:@*
dtype0
�
m/p_re_lu_19/alphaVarHandleOp*
_output_shapes
: *#

debug_namem/p_re_lu_19/alpha/*
dtype0*
shape:@*#
shared_namem/p_re_lu_19/alpha
u
&m/p_re_lu_19/alpha/Read/ReadVariableOpReadVariableOpm/p_re_lu_19/alpha*
_output_shapes
:@*
dtype0
�
v/batch_normalization_19/betaVarHandleOp*
_output_shapes
: *.

debug_name v/batch_normalization_19/beta/*
dtype0*
shape:@*.
shared_namev/batch_normalization_19/beta
�
1v/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpv/batch_normalization_19/beta*
_output_shapes
:@*
dtype0
�
m/batch_normalization_19/betaVarHandleOp*
_output_shapes
: *.

debug_name m/batch_normalization_19/beta/*
dtype0*
shape:@*.
shared_namem/batch_normalization_19/beta
�
1m/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpm/batch_normalization_19/beta*
_output_shapes
:@*
dtype0
�
v/batch_normalization_19/gammaVarHandleOp*
_output_shapes
: */

debug_name!v/batch_normalization_19/gamma/*
dtype0*
shape:@*/
shared_name v/batch_normalization_19/gamma
�
2v/batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpv/batch_normalization_19/gamma*
_output_shapes
:@*
dtype0
�
m/batch_normalization_19/gammaVarHandleOp*
_output_shapes
: */

debug_name!m/batch_normalization_19/gamma/*
dtype0*
shape:@*/
shared_name m/batch_normalization_19/gamma
�
2m/batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpm/batch_normalization_19/gamma*
_output_shapes
:@*
dtype0
�
v/dense_23/biasVarHandleOp*
_output_shapes
: * 

debug_namev/dense_23/bias/*
dtype0*
shape:@* 
shared_namev/dense_23/bias
o
#v/dense_23/bias/Read/ReadVariableOpReadVariableOpv/dense_23/bias*
_output_shapes
:@*
dtype0
�
m/dense_23/biasVarHandleOp*
_output_shapes
: * 

debug_namem/dense_23/bias/*
dtype0*
shape:@* 
shared_namem/dense_23/bias
o
#m/dense_23/bias/Read/ReadVariableOpReadVariableOpm/dense_23/bias*
_output_shapes
:@*
dtype0
�
v/dense_23/kernelVarHandleOp*
_output_shapes
: *"

debug_namev/dense_23/kernel/*
dtype0*
shape
:@@*"
shared_namev/dense_23/kernel
w
%v/dense_23/kernel/Read/ReadVariableOpReadVariableOpv/dense_23/kernel*
_output_shapes

:@@*
dtype0
�
m/dense_23/kernelVarHandleOp*
_output_shapes
: *"

debug_namem/dense_23/kernel/*
dtype0*
shape
:@@*"
shared_namem/dense_23/kernel
w
%m/dense_23/kernel/Read/ReadVariableOpReadVariableOpm/dense_23/kernel*
_output_shapes

:@@*
dtype0
�
v/p_re_lu_18/alphaVarHandleOp*
_output_shapes
: *#

debug_namev/p_re_lu_18/alpha/*
dtype0*
shape:@*#
shared_namev/p_re_lu_18/alpha
u
&v/p_re_lu_18/alpha/Read/ReadVariableOpReadVariableOpv/p_re_lu_18/alpha*
_output_shapes
:@*
dtype0
�
m/p_re_lu_18/alphaVarHandleOp*
_output_shapes
: *#

debug_namem/p_re_lu_18/alpha/*
dtype0*
shape:@*#
shared_namem/p_re_lu_18/alpha
u
&m/p_re_lu_18/alpha/Read/ReadVariableOpReadVariableOpm/p_re_lu_18/alpha*
_output_shapes
:@*
dtype0
�
v/batch_normalization_18/betaVarHandleOp*
_output_shapes
: *.

debug_name v/batch_normalization_18/beta/*
dtype0*
shape:@*.
shared_namev/batch_normalization_18/beta
�
1v/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpv/batch_normalization_18/beta*
_output_shapes
:@*
dtype0
�
m/batch_normalization_18/betaVarHandleOp*
_output_shapes
: *.

debug_name m/batch_normalization_18/beta/*
dtype0*
shape:@*.
shared_namem/batch_normalization_18/beta
�
1m/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpm/batch_normalization_18/beta*
_output_shapes
:@*
dtype0
�
v/batch_normalization_18/gammaVarHandleOp*
_output_shapes
: */

debug_name!v/batch_normalization_18/gamma/*
dtype0*
shape:@*/
shared_name v/batch_normalization_18/gamma
�
2v/batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpv/batch_normalization_18/gamma*
_output_shapes
:@*
dtype0
�
m/batch_normalization_18/gammaVarHandleOp*
_output_shapes
: */

debug_name!m/batch_normalization_18/gamma/*
dtype0*
shape:@*/
shared_name m/batch_normalization_18/gamma
�
2m/batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpm/batch_normalization_18/gamma*
_output_shapes
:@*
dtype0
�
v/dense_22/biasVarHandleOp*
_output_shapes
: * 

debug_namev/dense_22/bias/*
dtype0*
shape:@* 
shared_namev/dense_22/bias
o
#v/dense_22/bias/Read/ReadVariableOpReadVariableOpv/dense_22/bias*
_output_shapes
:@*
dtype0
�
m/dense_22/biasVarHandleOp*
_output_shapes
: * 

debug_namem/dense_22/bias/*
dtype0*
shape:@* 
shared_namem/dense_22/bias
o
#m/dense_22/bias/Read/ReadVariableOpReadVariableOpm/dense_22/bias*
_output_shapes
:@*
dtype0
�
v/dense_22/kernelVarHandleOp*
_output_shapes
: *"

debug_namev/dense_22/kernel/*
dtype0*
shape
:@@*"
shared_namev/dense_22/kernel
w
%v/dense_22/kernel/Read/ReadVariableOpReadVariableOpv/dense_22/kernel*
_output_shapes

:@@*
dtype0
�
m/dense_22/kernelVarHandleOp*
_output_shapes
: *"

debug_namem/dense_22/kernel/*
dtype0*
shape
:@@*"
shared_namem/dense_22/kernel
w
%m/dense_22/kernel/Read/ReadVariableOpReadVariableOpm/dense_22/kernel*
_output_shapes

:@@*
dtype0
�
v/p_re_lu_17/alphaVarHandleOp*
_output_shapes
: *#

debug_namev/p_re_lu_17/alpha/*
dtype0*
shape:@*#
shared_namev/p_re_lu_17/alpha
u
&v/p_re_lu_17/alpha/Read/ReadVariableOpReadVariableOpv/p_re_lu_17/alpha*
_output_shapes
:@*
dtype0
�
m/p_re_lu_17/alphaVarHandleOp*
_output_shapes
: *#

debug_namem/p_re_lu_17/alpha/*
dtype0*
shape:@*#
shared_namem/p_re_lu_17/alpha
u
&m/p_re_lu_17/alpha/Read/ReadVariableOpReadVariableOpm/p_re_lu_17/alpha*
_output_shapes
:@*
dtype0
�
v/batch_normalization_17/betaVarHandleOp*
_output_shapes
: *.

debug_name v/batch_normalization_17/beta/*
dtype0*
shape:@*.
shared_namev/batch_normalization_17/beta
�
1v/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpv/batch_normalization_17/beta*
_output_shapes
:@*
dtype0
�
m/batch_normalization_17/betaVarHandleOp*
_output_shapes
: *.

debug_name m/batch_normalization_17/beta/*
dtype0*
shape:@*.
shared_namem/batch_normalization_17/beta
�
1m/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpm/batch_normalization_17/beta*
_output_shapes
:@*
dtype0
�
v/batch_normalization_17/gammaVarHandleOp*
_output_shapes
: */

debug_name!v/batch_normalization_17/gamma/*
dtype0*
shape:@*/
shared_name v/batch_normalization_17/gamma
�
2v/batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpv/batch_normalization_17/gamma*
_output_shapes
:@*
dtype0
�
m/batch_normalization_17/gammaVarHandleOp*
_output_shapes
: */

debug_name!m/batch_normalization_17/gamma/*
dtype0*
shape:@*/
shared_name m/batch_normalization_17/gamma
�
2m/batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpm/batch_normalization_17/gamma*
_output_shapes
:@*
dtype0
�
v/dense_21/biasVarHandleOp*
_output_shapes
: * 

debug_namev/dense_21/bias/*
dtype0*
shape:@* 
shared_namev/dense_21/bias
o
#v/dense_21/bias/Read/ReadVariableOpReadVariableOpv/dense_21/bias*
_output_shapes
:@*
dtype0
�
m/dense_21/biasVarHandleOp*
_output_shapes
: * 

debug_namem/dense_21/bias/*
dtype0*
shape:@* 
shared_namem/dense_21/bias
o
#m/dense_21/bias/Read/ReadVariableOpReadVariableOpm/dense_21/bias*
_output_shapes
:@*
dtype0
�
v/dense_21/kernelVarHandleOp*
_output_shapes
: *"

debug_namev/dense_21/kernel/*
dtype0*
shape
:@@*"
shared_namev/dense_21/kernel
w
%v/dense_21/kernel/Read/ReadVariableOpReadVariableOpv/dense_21/kernel*
_output_shapes

:@@*
dtype0
�
m/dense_21/kernelVarHandleOp*
_output_shapes
: *"

debug_namem/dense_21/kernel/*
dtype0*
shape
:@@*"
shared_namem/dense_21/kernel
w
%m/dense_21/kernel/Read/ReadVariableOpReadVariableOpm/dense_21/kernel*
_output_shapes

:@@*
dtype0
�
v/p_re_lu_16/alphaVarHandleOp*
_output_shapes
: *#

debug_namev/p_re_lu_16/alpha/*
dtype0*
shape:@*#
shared_namev/p_re_lu_16/alpha
u
&v/p_re_lu_16/alpha/Read/ReadVariableOpReadVariableOpv/p_re_lu_16/alpha*
_output_shapes
:@*
dtype0
�
m/p_re_lu_16/alphaVarHandleOp*
_output_shapes
: *#

debug_namem/p_re_lu_16/alpha/*
dtype0*
shape:@*#
shared_namem/p_re_lu_16/alpha
u
&m/p_re_lu_16/alpha/Read/ReadVariableOpReadVariableOpm/p_re_lu_16/alpha*
_output_shapes
:@*
dtype0
�
v/batch_normalization_16/betaVarHandleOp*
_output_shapes
: *.

debug_name v/batch_normalization_16/beta/*
dtype0*
shape:@*.
shared_namev/batch_normalization_16/beta
�
1v/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpv/batch_normalization_16/beta*
_output_shapes
:@*
dtype0
�
m/batch_normalization_16/betaVarHandleOp*
_output_shapes
: *.

debug_name m/batch_normalization_16/beta/*
dtype0*
shape:@*.
shared_namem/batch_normalization_16/beta
�
1m/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpm/batch_normalization_16/beta*
_output_shapes
:@*
dtype0
�
v/batch_normalization_16/gammaVarHandleOp*
_output_shapes
: */

debug_name!v/batch_normalization_16/gamma/*
dtype0*
shape:@*/
shared_name v/batch_normalization_16/gamma
�
2v/batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpv/batch_normalization_16/gamma*
_output_shapes
:@*
dtype0
�
m/batch_normalization_16/gammaVarHandleOp*
_output_shapes
: */

debug_name!m/batch_normalization_16/gamma/*
dtype0*
shape:@*/
shared_name m/batch_normalization_16/gamma
�
2m/batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpm/batch_normalization_16/gamma*
_output_shapes
:@*
dtype0
�
v/dense_20/biasVarHandleOp*
_output_shapes
: * 

debug_namev/dense_20/bias/*
dtype0*
shape:@* 
shared_namev/dense_20/bias
o
#v/dense_20/bias/Read/ReadVariableOpReadVariableOpv/dense_20/bias*
_output_shapes
:@*
dtype0
�
m/dense_20/biasVarHandleOp*
_output_shapes
: * 

debug_namem/dense_20/bias/*
dtype0*
shape:@* 
shared_namem/dense_20/bias
o
#m/dense_20/bias/Read/ReadVariableOpReadVariableOpm/dense_20/bias*
_output_shapes
:@*
dtype0
�
v/dense_20/kernelVarHandleOp*
_output_shapes
: *"

debug_namev/dense_20/kernel/*
dtype0*
shape
:@*"
shared_namev/dense_20/kernel
w
%v/dense_20/kernel/Read/ReadVariableOpReadVariableOpv/dense_20/kernel*
_output_shapes

:@*
dtype0
�
m/dense_20/kernelVarHandleOp*
_output_shapes
: *"

debug_namem/dense_20/kernel/*
dtype0*
shape
:@*"
shared_namem/dense_20/kernel
w
%m/dense_20/kernel/Read/ReadVariableOpReadVariableOpm/dense_20/kernel*
_output_shapes

:@*
dtype0
�
current_learning_rateVarHandleOp*
_output_shapes
: *&

debug_namecurrent_learning_rate/*
dtype0*
shape: *&
shared_namecurrent_learning_rate
w
)current_learning_rate/Read/ReadVariableOpReadVariableOpcurrent_learning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_24/biasVarHandleOp*
_output_shapes
: *

debug_namedense_24/bias/*
dtype0*
shape:*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:*
dtype0
�
dense_24/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_24/kernel/*
dtype0*
shape
:@* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:@*
dtype0
�
p_re_lu_19/alphaVarHandleOp*
_output_shapes
: *!

debug_namep_re_lu_19/alpha/*
dtype0*
shape:@*!
shared_namep_re_lu_19/alpha
q
$p_re_lu_19/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_19/alpha*
_output_shapes
:@*
dtype0
�
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *7

debug_name)'batch_normalization_19/moving_variance/*
dtype0*
shape:@*7
shared_name(&batch_normalization_19/moving_variance
�
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes
:@*
dtype0
�
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *3

debug_name%#batch_normalization_19/moving_mean/*
dtype0*
shape:@*3
shared_name$"batch_normalization_19/moving_mean
�
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_19/beta/*
dtype0*
shape:@*,
shared_namebatch_normalization_19/beta
�
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_19/gamma/*
dtype0*
shape:@*-
shared_namebatch_normalization_19/gamma
�
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes
:@*
dtype0
�
dense_23/biasVarHandleOp*
_output_shapes
: *

debug_namedense_23/bias/*
dtype0*
shape:@*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:@*
dtype0
�
dense_23/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_23/kernel/*
dtype0*
shape
:@@* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:@@*
dtype0
�
p_re_lu_18/alphaVarHandleOp*
_output_shapes
: *!

debug_namep_re_lu_18/alpha/*
dtype0*
shape:@*!
shared_namep_re_lu_18/alpha
q
$p_re_lu_18/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_18/alpha*
_output_shapes
:@*
dtype0
�
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *7

debug_name)'batch_normalization_18/moving_variance/*
dtype0*
shape:@*7
shared_name(&batch_normalization_18/moving_variance
�
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes
:@*
dtype0
�
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *3

debug_name%#batch_normalization_18/moving_mean/*
dtype0*
shape:@*3
shared_name$"batch_normalization_18/moving_mean
�
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_18/beta/*
dtype0*
shape:@*,
shared_namebatch_normalization_18/beta
�
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_18/gamma/*
dtype0*
shape:@*-
shared_namebatch_normalization_18/gamma
�
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes
:@*
dtype0
�
dense_22/biasVarHandleOp*
_output_shapes
: *

debug_namedense_22/bias/*
dtype0*
shape:@*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:@*
dtype0
�
dense_22/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_22/kernel/*
dtype0*
shape
:@@* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:@@*
dtype0
�
p_re_lu_17/alphaVarHandleOp*
_output_shapes
: *!

debug_namep_re_lu_17/alpha/*
dtype0*
shape:@*!
shared_namep_re_lu_17/alpha
q
$p_re_lu_17/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_17/alpha*
_output_shapes
:@*
dtype0
�
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *7

debug_name)'batch_normalization_17/moving_variance/*
dtype0*
shape:@*7
shared_name(&batch_normalization_17/moving_variance
�
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:@*
dtype0
�
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *3

debug_name%#batch_normalization_17/moving_mean/*
dtype0*
shape:@*3
shared_name$"batch_normalization_17/moving_mean
�
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_17/beta/*
dtype0*
shape:@*,
shared_namebatch_normalization_17/beta
�
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_17/gamma/*
dtype0*
shape:@*-
shared_namebatch_normalization_17/gamma
�
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:@*
dtype0
�
dense_21/biasVarHandleOp*
_output_shapes
: *

debug_namedense_21/bias/*
dtype0*
shape:@*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:@*
dtype0
�
dense_21/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_21/kernel/*
dtype0*
shape
:@@* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:@@*
dtype0
�
p_re_lu_16/alphaVarHandleOp*
_output_shapes
: *!

debug_namep_re_lu_16/alpha/*
dtype0*
shape:@*!
shared_namep_re_lu_16/alpha
q
$p_re_lu_16/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_16/alpha*
_output_shapes
:@*
dtype0
�
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *7

debug_name)'batch_normalization_16/moving_variance/*
dtype0*
shape:@*7
shared_name(&batch_normalization_16/moving_variance
�
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:@*
dtype0
�
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *3

debug_name%#batch_normalization_16/moving_mean/*
dtype0*
shape:@*3
shared_name$"batch_normalization_16/moving_mean
�
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_16/beta/*
dtype0*
shape:@*,
shared_namebatch_normalization_16/beta
�
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_16/gamma/*
dtype0*
shape:@*-
shared_namebatch_normalization_16/gamma
�
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:@*
dtype0
�
dense_20/biasVarHandleOp*
_output_shapes
: *

debug_namedense_20/bias/*
dtype0*
shape:@*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:@*
dtype0
�
dense_20/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_20/kernel/*
dtype0*
shape
:@* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:@*
dtype0
�
serving_default_dense_20_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_20_inputdense_20/kerneldense_20/bias&batch_normalization_16/moving_variancebatch_normalization_16/gamma"batch_normalization_16/moving_meanbatch_normalization_16/betap_re_lu_16/alphadense_21/kerneldense_21/bias&batch_normalization_17/moving_variancebatch_normalization_17/gamma"batch_normalization_17/moving_meanbatch_normalization_17/betap_re_lu_17/alphadense_22/kerneldense_22/bias&batch_normalization_18/moving_variancebatch_normalization_18/gamma"batch_normalization_18/moving_meanbatch_normalization_18/betap_re_lu_18/alphadense_23/kerneldense_23/bias&batch_normalization_19/moving_variancebatch_normalization_19/gamma"batch_normalization_19/moving_meanbatch_normalization_19/betap_re_lu_19/alphadense_24/kerneldense_24/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_95203

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
layer_with_weights-11
layer-11
layer_with_weights-12
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%axis
	&gamma
'beta
(moving_mean
)moving_variance*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
	0alpha*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance*
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
	Jalpha*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
	dalpha*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias*
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance*
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
	~alpha*
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
0
1
&2
'3
(4
)5
06
77
88
@9
A10
B11
C12
J13
Q14
R15
Z16
[17
\18
]19
d20
k21
l22
t23
u24
v25
w26
~27
�28
�29*
�
0
1
&2
'3
04
75
86
@7
A8
J9
Q10
R11
Z12
[13
d14
k15
l16
t17
u18
~19
�20
�21*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_current_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_20/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
&0
'1
(2
)3*

&0
'1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

00*

00*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEp_re_lu_16/alpha5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
@0
A1
B2
C3*

@0
A1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

J0*

J0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEp_re_lu_17/alpha5layer_with_weights-5/alpha/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
Z0
[1
\2
]3*

Z0
[1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

d0*

d0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEp_re_lu_18/alpha5layer_with_weights-8/alpha/.ATTRIBUTES/VARIABLE_VALUE*

k0
l1*

k0
l1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
t0
u1
v2
w3*

t0
u1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_19/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_19/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_19/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE&batch_normalization_19/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

~0*

~0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEp_re_lu_19/alpha6layer_with_weights-11/alpha/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_24/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_24/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
<
(0
)1
B2
C3
\4
]5
v6
w7*
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*

�0
�1*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcurrent_learning_rate;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21*
* 
* 
* 
* 
* 
* 
* 
* 
* 

(0
)1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

B0
C1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

\0
]1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

v0
w1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
\V
VARIABLE_VALUEm/dense_20/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEv/dense_20/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEm/dense_20/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEv/dense_20/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEm/batch_normalization_16/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEv/batch_normalization_16/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEm/batch_normalization_16/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEv/batch_normalization_16/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEm/p_re_lu_16/alpha1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEv/p_re_lu_16/alpha2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEm/dense_21/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEv/dense_21/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEm/dense_21/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEv/dense_21/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEm/batch_normalization_17/gamma2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEv/batch_normalization_17/gamma2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEm/batch_normalization_17/beta2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEv/batch_normalization_17/beta2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEm/p_re_lu_17/alpha2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEv/p_re_lu_17/alpha2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEm/dense_22/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEv/dense_22/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEm/dense_22/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEv/dense_22/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEm/batch_normalization_18/gamma2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEv/batch_normalization_18/gamma2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEm/batch_normalization_18/beta2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEv/batch_normalization_18/beta2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEm/p_re_lu_18/alpha2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEv/p_re_lu_18/alpha2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEm/dense_23/kernel2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEv/dense_23/kernel2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEm/dense_23/bias2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEv/dense_23/bias2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEm/batch_normalization_19/gamma2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEv/batch_normalization_19/gamma2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEm/batch_normalization_19/beta2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEv/batch_normalization_19/beta2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEm/p_re_lu_19/alpha2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEv/p_re_lu_19/alpha2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEm/dense_24/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEv/dense_24/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEm/dense_24/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEv/dense_24/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variancep_re_lu_16/alphadense_21/kerneldense_21/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancep_re_lu_17/alphadense_22/kerneldense_22/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_variancep_re_lu_18/alphadense_23/kerneldense_23/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancep_re_lu_19/alphadense_24/kerneldense_24/bias	iterationcurrent_learning_ratem/dense_20/kernelv/dense_20/kernelm/dense_20/biasv/dense_20/biasm/batch_normalization_16/gammav/batch_normalization_16/gammam/batch_normalization_16/betav/batch_normalization_16/betam/p_re_lu_16/alphav/p_re_lu_16/alpham/dense_21/kernelv/dense_21/kernelm/dense_21/biasv/dense_21/biasm/batch_normalization_17/gammav/batch_normalization_17/gammam/batch_normalization_17/betav/batch_normalization_17/betam/p_re_lu_17/alphav/p_re_lu_17/alpham/dense_22/kernelv/dense_22/kernelm/dense_22/biasv/dense_22/biasm/batch_normalization_18/gammav/batch_normalization_18/gammam/batch_normalization_18/betav/batch_normalization_18/betam/p_re_lu_18/alphav/p_re_lu_18/alpham/dense_23/kernelv/dense_23/kernelm/dense_23/biasv/dense_23/biasm/batch_normalization_19/gammav/batch_normalization_19/gammam/batch_normalization_19/betav/batch_normalization_19/betam/p_re_lu_19/alphav/p_re_lu_19/alpham/dense_24/kernelv/dense_24/kernelm/dense_24/biasv/dense_24/biastotal_1count_1totalcountConst*]
TinV
T2R*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_96197
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variancep_re_lu_16/alphadense_21/kerneldense_21/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancep_re_lu_17/alphadense_22/kerneldense_22/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_variancep_re_lu_18/alphadense_23/kerneldense_23/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancep_re_lu_19/alphadense_24/kerneldense_24/bias	iterationcurrent_learning_ratem/dense_20/kernelv/dense_20/kernelm/dense_20/biasv/dense_20/biasm/batch_normalization_16/gammav/batch_normalization_16/gammam/batch_normalization_16/betav/batch_normalization_16/betam/p_re_lu_16/alphav/p_re_lu_16/alpham/dense_21/kernelv/dense_21/kernelm/dense_21/biasv/dense_21/biasm/batch_normalization_17/gammav/batch_normalization_17/gammam/batch_normalization_17/betav/batch_normalization_17/betam/p_re_lu_17/alphav/p_re_lu_17/alpham/dense_22/kernelv/dense_22/kernelm/dense_22/biasv/dense_22/biasm/batch_normalization_18/gammav/batch_normalization_18/gammam/batch_normalization_18/betav/batch_normalization_18/betam/p_re_lu_18/alphav/p_re_lu_18/alpham/dense_23/kernelv/dense_23/kernelm/dense_23/biasv/dense_23/biasm/batch_normalization_19/gammav/batch_normalization_19/gammam/batch_normalization_19/betav/batch_normalization_19/betam/p_re_lu_19/alphav/p_re_lu_19/alpham/dense_24/kernelv/dense_24/kernelm/dense_24/biasv/dense_24/biastotal_1count_1totalcount*\
TinU
S2Q*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_96446ԭ
�
z
*__inference_p_re_lu_18_layer_call_fn_95545

inputs
unknown:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_18_layer_call_and_return_conditional_losses_94651o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95541:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�&
�
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_94593

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�S
�
G__inference_sequential_4_layer_call_and_return_conditional_losses_94885
dense_20_input 
dense_20_94770:@
dense_20_94772:@*
batch_normalization_16_94775:@*
batch_normalization_16_94777:@*
batch_normalization_16_94779:@*
batch_normalization_16_94781:@
p_re_lu_16_94784:@ 
dense_21_94797:@@
dense_21_94799:@*
batch_normalization_17_94802:@*
batch_normalization_17_94804:@*
batch_normalization_17_94806:@*
batch_normalization_17_94808:@
p_re_lu_17_94811:@ 
dense_22_94824:@@
dense_22_94826:@*
batch_normalization_18_94829:@*
batch_normalization_18_94831:@*
batch_normalization_18_94833:@*
batch_normalization_18_94835:@
p_re_lu_18_94838:@ 
dense_23_94851:@@
dense_23_94853:@*
batch_normalization_19_94856:@*
batch_normalization_19_94858:@*
batch_normalization_19_94860:@*
batch_normalization_19_94862:@
p_re_lu_19_94865:@ 
dense_24_94879:@
dense_24_94881:
identity��.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�"p_re_lu_16/StatefulPartitionedCall�"p_re_lu_17/StatefulPartitionedCall�"p_re_lu_18/StatefulPartitionedCall�"p_re_lu_19/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_inputdense_20_94770dense_20_94772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_94769�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_16_94775batch_normalization_16_94777batch_normalization_16_94779batch_normalization_16_94781*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_94395�
"p_re_lu_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0p_re_lu_16_94784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_16_layer_call_and_return_conditional_losses_94453�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall+p_re_lu_16/StatefulPartitionedCall:output:0dense_21_94797dense_21_94799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_94796�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_17_94802batch_normalization_17_94804batch_normalization_17_94806batch_normalization_17_94808*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_94494�
"p_re_lu_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0p_re_lu_17_94811*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_17_layer_call_and_return_conditional_losses_94552�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+p_re_lu_17/StatefulPartitionedCall:output:0dense_22_94824dense_22_94826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_94823�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_18_94829batch_normalization_18_94831batch_normalization_18_94833batch_normalization_18_94835*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_94593�
"p_re_lu_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0p_re_lu_18_94838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_18_layer_call_and_return_conditional_losses_94651�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+p_re_lu_18/StatefulPartitionedCall:output:0dense_23_94851dense_23_94853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_94850�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_19_94856batch_normalization_19_94858batch_normalization_19_94860batch_normalization_19_94862*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_94692�
"p_re_lu_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0p_re_lu_19_94865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_19_layer_call_and_return_conditional_losses_94750�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+p_re_lu_19/StatefulPartitionedCall:output:0dense_24_94879dense_24_94881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_94878x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^p_re_lu_16/StatefulPartitionedCall#^p_re_lu_17/StatefulPartitionedCall#^p_re_lu_18/StatefulPartitionedCall#^p_re_lu_19/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"p_re_lu_16/StatefulPartitionedCall"p_re_lu_16/StatefulPartitionedCall2H
"p_re_lu_17/StatefulPartitionedCall"p_re_lu_17/StatefulPartitionedCall2H
"p_re_lu_18/StatefulPartitionedCall"p_re_lu_18/StatefulPartitionedCall2H
"p_re_lu_19/StatefulPartitionedCall"p_re_lu_19/StatefulPartitionedCall:%!

_user_specified_name94881:%!

_user_specified_name94879:%!

_user_specified_name94865:%!

_user_specified_name94862:%!

_user_specified_name94860:%!

_user_specified_name94858:%!

_user_specified_name94856:%!

_user_specified_name94853:%!

_user_specified_name94851:%!

_user_specified_name94838:%!

_user_specified_name94835:%!

_user_specified_name94833:%!

_user_specified_name94831:%!

_user_specified_name94829:%!

_user_specified_name94826:%!

_user_specified_name94824:%!

_user_specified_name94811:%!

_user_specified_name94808:%!

_user_specified_name94806:%!

_user_specified_name94804:%
!

_user_specified_name94802:%	!

_user_specified_name94799:%!

_user_specified_name94797:%!

_user_specified_name94784:%!

_user_specified_name94781:%!

_user_specified_name94779:%!

_user_specified_name94777:%!

_user_specified_name94775:%!

_user_specified_name94772:%!

_user_specified_name94770:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�&
�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_95400

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
E__inference_p_re_lu_16_layer_call_and_return_conditional_losses_95321

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_17_layer_call_fn_95353

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_94494o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95349:%!

_user_specified_name95347:%!

_user_specified_name95345:%!

_user_specified_name95343:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
z
*__inference_p_re_lu_16_layer_call_fn_95309

inputs
unknown:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_16_layer_call_and_return_conditional_losses_94453o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95305:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_19_layer_call_fn_95589

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_94692o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95585:%!

_user_specified_name95583:%!

_user_specified_name95581:%!

_user_specified_name95579:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_95420

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_95538

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_24_layer_call_and_return_conditional_losses_95695

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_95203
dense_20_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_94361o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95199:%!

_user_specified_name95197:%!

_user_specified_name95195:%!

_user_specified_name95193:%!

_user_specified_name95191:%!

_user_specified_name95189:%!

_user_specified_name95187:%!

_user_specified_name95185:%!

_user_specified_name95183:%!

_user_specified_name95181:%!

_user_specified_name95179:%!

_user_specified_name95177:%!

_user_specified_name95175:%!

_user_specified_name95173:%!

_user_specified_name95171:%!

_user_specified_name95169:%!

_user_specified_name95167:%!

_user_specified_name95165:%!

_user_specified_name95163:%!

_user_specified_name95161:%
!

_user_specified_name95159:%	!

_user_specified_name95157:%!

_user_specified_name95155:%!

_user_specified_name95153:%!

_user_specified_name95151:%!

_user_specified_name95149:%!

_user_specified_name95147:%!

_user_specified_name95145:%!

_user_specified_name95143:%!

_user_specified_name95141:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�	
�
C__inference_dense_22_layer_call_and_return_conditional_losses_94823

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_dense_22_layer_call_and_return_conditional_losses_95458

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_94361
dense_20_inputF
4sequential_4_dense_20_matmul_readvariableop_resource:@C
5sequential_4_dense_20_biasadd_readvariableop_resource:@S
Esequential_4_batch_normalization_16_batchnorm_readvariableop_resource:@W
Isequential_4_batch_normalization_16_batchnorm_mul_readvariableop_resource:@U
Gsequential_4_batch_normalization_16_batchnorm_readvariableop_1_resource:@U
Gsequential_4_batch_normalization_16_batchnorm_readvariableop_2_resource:@=
/sequential_4_p_re_lu_16_readvariableop_resource:@F
4sequential_4_dense_21_matmul_readvariableop_resource:@@C
5sequential_4_dense_21_biasadd_readvariableop_resource:@S
Esequential_4_batch_normalization_17_batchnorm_readvariableop_resource:@W
Isequential_4_batch_normalization_17_batchnorm_mul_readvariableop_resource:@U
Gsequential_4_batch_normalization_17_batchnorm_readvariableop_1_resource:@U
Gsequential_4_batch_normalization_17_batchnorm_readvariableop_2_resource:@=
/sequential_4_p_re_lu_17_readvariableop_resource:@F
4sequential_4_dense_22_matmul_readvariableop_resource:@@C
5sequential_4_dense_22_biasadd_readvariableop_resource:@S
Esequential_4_batch_normalization_18_batchnorm_readvariableop_resource:@W
Isequential_4_batch_normalization_18_batchnorm_mul_readvariableop_resource:@U
Gsequential_4_batch_normalization_18_batchnorm_readvariableop_1_resource:@U
Gsequential_4_batch_normalization_18_batchnorm_readvariableop_2_resource:@=
/sequential_4_p_re_lu_18_readvariableop_resource:@F
4sequential_4_dense_23_matmul_readvariableop_resource:@@C
5sequential_4_dense_23_biasadd_readvariableop_resource:@S
Esequential_4_batch_normalization_19_batchnorm_readvariableop_resource:@W
Isequential_4_batch_normalization_19_batchnorm_mul_readvariableop_resource:@U
Gsequential_4_batch_normalization_19_batchnorm_readvariableop_1_resource:@U
Gsequential_4_batch_normalization_19_batchnorm_readvariableop_2_resource:@=
/sequential_4_p_re_lu_19_readvariableop_resource:@F
4sequential_4_dense_24_matmul_readvariableop_resource:@C
5sequential_4_dense_24_biasadd_readvariableop_resource:
identity��<sequential_4/batch_normalization_16/batchnorm/ReadVariableOp�>sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_1�>sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_2�@sequential_4/batch_normalization_16/batchnorm/mul/ReadVariableOp�<sequential_4/batch_normalization_17/batchnorm/ReadVariableOp�>sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_1�>sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_2�@sequential_4/batch_normalization_17/batchnorm/mul/ReadVariableOp�<sequential_4/batch_normalization_18/batchnorm/ReadVariableOp�>sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_1�>sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_2�@sequential_4/batch_normalization_18/batchnorm/mul/ReadVariableOp�<sequential_4/batch_normalization_19/batchnorm/ReadVariableOp�>sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_1�>sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_2�@sequential_4/batch_normalization_19/batchnorm/mul/ReadVariableOp�,sequential_4/dense_20/BiasAdd/ReadVariableOp�+sequential_4/dense_20/MatMul/ReadVariableOp�,sequential_4/dense_21/BiasAdd/ReadVariableOp�+sequential_4/dense_21/MatMul/ReadVariableOp�,sequential_4/dense_22/BiasAdd/ReadVariableOp�+sequential_4/dense_22/MatMul/ReadVariableOp�,sequential_4/dense_23/BiasAdd/ReadVariableOp�+sequential_4/dense_23/MatMul/ReadVariableOp�,sequential_4/dense_24/BiasAdd/ReadVariableOp�+sequential_4/dense_24/MatMul/ReadVariableOp�&sequential_4/p_re_lu_16/ReadVariableOp�&sequential_4/p_re_lu_17/ReadVariableOp�&sequential_4/p_re_lu_18/ReadVariableOp�&sequential_4/p_re_lu_19/ReadVariableOp�
+sequential_4/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_20_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
sequential_4/dense_20/MatMulMatMuldense_20_input3sequential_4/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_4/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_4/dense_20/BiasAddBiasAdd&sequential_4/dense_20/MatMul:product:04sequential_4/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<sequential_4/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0x
3sequential_4/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_4/batch_normalization_16/batchnorm/addAddV2Dsequential_4/batch_normalization_16/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_16/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:@�
@sequential_4/batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
1sequential_4/batch_normalization_16/batchnorm/mulMul7sequential_4/batch_normalization_16/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_16/batchnorm/mul_1Mul&sequential_4/dense_20/BiasAdd:output:05sequential_4/batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
>sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3sequential_4/batch_normalization_16/batchnorm/mul_2MulFsequential_4/batch_normalization_16/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
>sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
1sequential_4/batch_normalization_16/batchnorm/subSubFsequential_4/batch_normalization_16/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_16/batchnorm/add_1AddV27sequential_4/batch_normalization_16/batchnorm/mul_1:z:05sequential_4/batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_16/ReluRelu7sequential_4/batch_normalization_16/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
&sequential_4/p_re_lu_16/ReadVariableOpReadVariableOp/sequential_4_p_re_lu_16_readvariableop_resource*
_output_shapes
:@*
dtype0w
sequential_4/p_re_lu_16/NegNeg.sequential_4/p_re_lu_16/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
sequential_4/p_re_lu_16/Neg_1Neg7sequential_4/batch_normalization_16/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@{
sequential_4/p_re_lu_16/Relu_1Relu!sequential_4/p_re_lu_16/Neg_1:y:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_16/mulMulsequential_4/p_re_lu_16/Neg:y:0,sequential_4/p_re_lu_16/Relu_1:activations:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_16/addAddV2*sequential_4/p_re_lu_16/Relu:activations:0sequential_4/p_re_lu_16/mul:z:0*
T0*'
_output_shapes
:���������@�
+sequential_4/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_21_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
sequential_4/dense_21/MatMulMatMulsequential_4/p_re_lu_16/add:z:03sequential_4/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_4/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_4/dense_21/BiasAddBiasAdd&sequential_4/dense_21/MatMul:product:04sequential_4/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<sequential_4/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0x
3sequential_4/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_4/batch_normalization_17/batchnorm/addAddV2Dsequential_4/batch_normalization_17/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_17/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
:@�
@sequential_4/batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
1sequential_4/batch_normalization_17/batchnorm/mulMul7sequential_4/batch_normalization_17/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_17/batchnorm/mul_1Mul&sequential_4/dense_21/BiasAdd:output:05sequential_4/batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
>sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3sequential_4/batch_normalization_17/batchnorm/mul_2MulFsequential_4/batch_normalization_17/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
>sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
1sequential_4/batch_normalization_17/batchnorm/subSubFsequential_4/batch_normalization_17/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_17/batchnorm/add_1AddV27sequential_4/batch_normalization_17/batchnorm/mul_1:z:05sequential_4/batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_17/ReluRelu7sequential_4/batch_normalization_17/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
&sequential_4/p_re_lu_17/ReadVariableOpReadVariableOp/sequential_4_p_re_lu_17_readvariableop_resource*
_output_shapes
:@*
dtype0w
sequential_4/p_re_lu_17/NegNeg.sequential_4/p_re_lu_17/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
sequential_4/p_re_lu_17/Neg_1Neg7sequential_4/batch_normalization_17/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@{
sequential_4/p_re_lu_17/Relu_1Relu!sequential_4/p_re_lu_17/Neg_1:y:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_17/mulMulsequential_4/p_re_lu_17/Neg:y:0,sequential_4/p_re_lu_17/Relu_1:activations:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_17/addAddV2*sequential_4/p_re_lu_17/Relu:activations:0sequential_4/p_re_lu_17/mul:z:0*
T0*'
_output_shapes
:���������@�
+sequential_4/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_22_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
sequential_4/dense_22/MatMulMatMulsequential_4/p_re_lu_17/add:z:03sequential_4/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_4/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_4/dense_22/BiasAddBiasAdd&sequential_4/dense_22/MatMul:product:04sequential_4/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<sequential_4/batch_normalization_18/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_18_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0x
3sequential_4/batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_4/batch_normalization_18/batchnorm/addAddV2Dsequential_4/batch_normalization_18/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_18/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes
:@�
@sequential_4/batch_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
1sequential_4/batch_normalization_18/batchnorm/mulMul7sequential_4/batch_normalization_18/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_18/batchnorm/mul_1Mul&sequential_4/dense_22/BiasAdd:output:05sequential_4/batch_normalization_18/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
>sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_18_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3sequential_4/batch_normalization_18/batchnorm/mul_2MulFsequential_4/batch_normalization_18/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
>sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_18_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
1sequential_4/batch_normalization_18/batchnorm/subSubFsequential_4/batch_normalization_18/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_18/batchnorm/add_1AddV27sequential_4/batch_normalization_18/batchnorm/mul_1:z:05sequential_4/batch_normalization_18/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_18/ReluRelu7sequential_4/batch_normalization_18/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
&sequential_4/p_re_lu_18/ReadVariableOpReadVariableOp/sequential_4_p_re_lu_18_readvariableop_resource*
_output_shapes
:@*
dtype0w
sequential_4/p_re_lu_18/NegNeg.sequential_4/p_re_lu_18/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
sequential_4/p_re_lu_18/Neg_1Neg7sequential_4/batch_normalization_18/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@{
sequential_4/p_re_lu_18/Relu_1Relu!sequential_4/p_re_lu_18/Neg_1:y:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_18/mulMulsequential_4/p_re_lu_18/Neg:y:0,sequential_4/p_re_lu_18/Relu_1:activations:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_18/addAddV2*sequential_4/p_re_lu_18/Relu:activations:0sequential_4/p_re_lu_18/mul:z:0*
T0*'
_output_shapes
:���������@�
+sequential_4/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_23_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
sequential_4/dense_23/MatMulMatMulsequential_4/p_re_lu_18/add:z:03sequential_4/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_4/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_4/dense_23/BiasAddBiasAdd&sequential_4/dense_23/MatMul:product:04sequential_4/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<sequential_4/batch_normalization_19/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_19_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0x
3sequential_4/batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_4/batch_normalization_19/batchnorm/addAddV2Dsequential_4/batch_normalization_19/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_19/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes
:@�
@sequential_4/batch_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
1sequential_4/batch_normalization_19/batchnorm/mulMul7sequential_4/batch_normalization_19/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_19/batchnorm/mul_1Mul&sequential_4/dense_23/BiasAdd:output:05sequential_4/batch_normalization_19/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
>sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_19_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3sequential_4/batch_normalization_19/batchnorm/mul_2MulFsequential_4/batch_normalization_19/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
>sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_19_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
1sequential_4/batch_normalization_19/batchnorm/subSubFsequential_4/batch_normalization_19/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
3sequential_4/batch_normalization_19/batchnorm/add_1AddV27sequential_4/batch_normalization_19/batchnorm/mul_1:z:05sequential_4/batch_normalization_19/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_19/ReluRelu7sequential_4/batch_normalization_19/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
&sequential_4/p_re_lu_19/ReadVariableOpReadVariableOp/sequential_4_p_re_lu_19_readvariableop_resource*
_output_shapes
:@*
dtype0w
sequential_4/p_re_lu_19/NegNeg.sequential_4/p_re_lu_19/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
sequential_4/p_re_lu_19/Neg_1Neg7sequential_4/batch_normalization_19/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@{
sequential_4/p_re_lu_19/Relu_1Relu!sequential_4/p_re_lu_19/Neg_1:y:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_19/mulMulsequential_4/p_re_lu_19/Neg:y:0,sequential_4/p_re_lu_19/Relu_1:activations:0*
T0*'
_output_shapes
:���������@�
sequential_4/p_re_lu_19/addAddV2*sequential_4/p_re_lu_19/Relu:activations:0sequential_4/p_re_lu_19/mul:z:0*
T0*'
_output_shapes
:���������@�
+sequential_4/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_24_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
sequential_4/dense_24/MatMulMatMulsequential_4/p_re_lu_19/add:z:03sequential_4/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_4/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_4/dense_24/BiasAddBiasAdd&sequential_4/dense_24/MatMul:product:04sequential_4/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_4/dense_24/SigmoidSigmoid&sequential_4/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:���������p
IdentityIdentity!sequential_4/dense_24/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp=^sequential_4/batch_normalization_16/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_16/batchnorm/mul/ReadVariableOp=^sequential_4/batch_normalization_17/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_17/batchnorm/mul/ReadVariableOp=^sequential_4/batch_normalization_18/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_18/batchnorm/mul/ReadVariableOp=^sequential_4/batch_normalization_19/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_19/batchnorm/mul/ReadVariableOp-^sequential_4/dense_20/BiasAdd/ReadVariableOp,^sequential_4/dense_20/MatMul/ReadVariableOp-^sequential_4/dense_21/BiasAdd/ReadVariableOp,^sequential_4/dense_21/MatMul/ReadVariableOp-^sequential_4/dense_22/BiasAdd/ReadVariableOp,^sequential_4/dense_22/MatMul/ReadVariableOp-^sequential_4/dense_23/BiasAdd/ReadVariableOp,^sequential_4/dense_23/MatMul/ReadVariableOp-^sequential_4/dense_24/BiasAdd/ReadVariableOp,^sequential_4/dense_24/MatMul/ReadVariableOp'^sequential_4/p_re_lu_16/ReadVariableOp'^sequential_4/p_re_lu_17/ReadVariableOp'^sequential_4/p_re_lu_18/ReadVariableOp'^sequential_4/p_re_lu_19/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2�
>sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_12�
>sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_16/batchnorm/ReadVariableOp_22|
<sequential_4/batch_normalization_16/batchnorm/ReadVariableOp<sequential_4/batch_normalization_16/batchnorm/ReadVariableOp2�
@sequential_4/batch_normalization_16/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_16/batchnorm/mul/ReadVariableOp2�
>sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_12�
>sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_17/batchnorm/ReadVariableOp_22|
<sequential_4/batch_normalization_17/batchnorm/ReadVariableOp<sequential_4/batch_normalization_17/batchnorm/ReadVariableOp2�
@sequential_4/batch_normalization_17/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_17/batchnorm/mul/ReadVariableOp2�
>sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_12�
>sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_18/batchnorm/ReadVariableOp_22|
<sequential_4/batch_normalization_18/batchnorm/ReadVariableOp<sequential_4/batch_normalization_18/batchnorm/ReadVariableOp2�
@sequential_4/batch_normalization_18/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_18/batchnorm/mul/ReadVariableOp2�
>sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_12�
>sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_19/batchnorm/ReadVariableOp_22|
<sequential_4/batch_normalization_19/batchnorm/ReadVariableOp<sequential_4/batch_normalization_19/batchnorm/ReadVariableOp2�
@sequential_4/batch_normalization_19/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_19/batchnorm/mul/ReadVariableOp2\
,sequential_4/dense_20/BiasAdd/ReadVariableOp,sequential_4/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_20/MatMul/ReadVariableOp+sequential_4/dense_20/MatMul/ReadVariableOp2\
,sequential_4/dense_21/BiasAdd/ReadVariableOp,sequential_4/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_21/MatMul/ReadVariableOp+sequential_4/dense_21/MatMul/ReadVariableOp2\
,sequential_4/dense_22/BiasAdd/ReadVariableOp,sequential_4/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_22/MatMul/ReadVariableOp+sequential_4/dense_22/MatMul/ReadVariableOp2\
,sequential_4/dense_23/BiasAdd/ReadVariableOp,sequential_4/dense_23/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_23/MatMul/ReadVariableOp+sequential_4/dense_23/MatMul/ReadVariableOp2\
,sequential_4/dense_24/BiasAdd/ReadVariableOp,sequential_4/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_24/MatMul/ReadVariableOp+sequential_4/dense_24/MatMul/ReadVariableOp2P
&sequential_4/p_re_lu_16/ReadVariableOp&sequential_4/p_re_lu_16/ReadVariableOp2P
&sequential_4/p_re_lu_17/ReadVariableOp&sequential_4/p_re_lu_17/ReadVariableOp2P
&sequential_4/p_re_lu_18/ReadVariableOp&sequential_4/p_re_lu_18/ReadVariableOp2P
&sequential_4/p_re_lu_19/ReadVariableOp&sequential_4/p_re_lu_19/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�	
�
E__inference_p_re_lu_19_layer_call_and_return_conditional_losses_94750

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�

�
C__inference_dense_24_layer_call_and_return_conditional_losses_94878

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_dense_21_layer_call_and_return_conditional_losses_94796

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_dense_21_layer_call_and_return_conditional_losses_95340

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_dense_23_layer_call_and_return_conditional_losses_94850

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_95302

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_24_layer_call_fn_95684

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_94878o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95680:%!

_user_specified_name95678:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_19_layer_call_fn_95602

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_94712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95598:%!

_user_specified_name95596:%!

_user_specified_name95594:%!

_user_specified_name95592:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_22_layer_call_fn_95448

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_94823o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95444:%!

_user_specified_name95442:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_dense_20_layer_call_and_return_conditional_losses_95222

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�2
!__inference__traced_restore_96446
file_prefix2
 assignvariableop_dense_20_kernel:@.
 assignvariableop_1_dense_20_bias:@=
/assignvariableop_2_batch_normalization_16_gamma:@<
.assignvariableop_3_batch_normalization_16_beta:@C
5assignvariableop_4_batch_normalization_16_moving_mean:@G
9assignvariableop_5_batch_normalization_16_moving_variance:@1
#assignvariableop_6_p_re_lu_16_alpha:@4
"assignvariableop_7_dense_21_kernel:@@.
 assignvariableop_8_dense_21_bias:@=
/assignvariableop_9_batch_normalization_17_gamma:@=
/assignvariableop_10_batch_normalization_17_beta:@D
6assignvariableop_11_batch_normalization_17_moving_mean:@H
:assignvariableop_12_batch_normalization_17_moving_variance:@2
$assignvariableop_13_p_re_lu_17_alpha:@5
#assignvariableop_14_dense_22_kernel:@@/
!assignvariableop_15_dense_22_bias:@>
0assignvariableop_16_batch_normalization_18_gamma:@=
/assignvariableop_17_batch_normalization_18_beta:@D
6assignvariableop_18_batch_normalization_18_moving_mean:@H
:assignvariableop_19_batch_normalization_18_moving_variance:@2
$assignvariableop_20_p_re_lu_18_alpha:@5
#assignvariableop_21_dense_23_kernel:@@/
!assignvariableop_22_dense_23_bias:@>
0assignvariableop_23_batch_normalization_19_gamma:@=
/assignvariableop_24_batch_normalization_19_beta:@D
6assignvariableop_25_batch_normalization_19_moving_mean:@H
:assignvariableop_26_batch_normalization_19_moving_variance:@2
$assignvariableop_27_p_re_lu_19_alpha:@5
#assignvariableop_28_dense_24_kernel:@/
!assignvariableop_29_dense_24_bias:'
assignvariableop_30_iteration:	 3
)assignvariableop_31_current_learning_rate: 7
%assignvariableop_32_m_dense_20_kernel:@7
%assignvariableop_33_v_dense_20_kernel:@1
#assignvariableop_34_m_dense_20_bias:@1
#assignvariableop_35_v_dense_20_bias:@@
2assignvariableop_36_m_batch_normalization_16_gamma:@@
2assignvariableop_37_v_batch_normalization_16_gamma:@?
1assignvariableop_38_m_batch_normalization_16_beta:@?
1assignvariableop_39_v_batch_normalization_16_beta:@4
&assignvariableop_40_m_p_re_lu_16_alpha:@4
&assignvariableop_41_v_p_re_lu_16_alpha:@7
%assignvariableop_42_m_dense_21_kernel:@@7
%assignvariableop_43_v_dense_21_kernel:@@1
#assignvariableop_44_m_dense_21_bias:@1
#assignvariableop_45_v_dense_21_bias:@@
2assignvariableop_46_m_batch_normalization_17_gamma:@@
2assignvariableop_47_v_batch_normalization_17_gamma:@?
1assignvariableop_48_m_batch_normalization_17_beta:@?
1assignvariableop_49_v_batch_normalization_17_beta:@4
&assignvariableop_50_m_p_re_lu_17_alpha:@4
&assignvariableop_51_v_p_re_lu_17_alpha:@7
%assignvariableop_52_m_dense_22_kernel:@@7
%assignvariableop_53_v_dense_22_kernel:@@1
#assignvariableop_54_m_dense_22_bias:@1
#assignvariableop_55_v_dense_22_bias:@@
2assignvariableop_56_m_batch_normalization_18_gamma:@@
2assignvariableop_57_v_batch_normalization_18_gamma:@?
1assignvariableop_58_m_batch_normalization_18_beta:@?
1assignvariableop_59_v_batch_normalization_18_beta:@4
&assignvariableop_60_m_p_re_lu_18_alpha:@4
&assignvariableop_61_v_p_re_lu_18_alpha:@7
%assignvariableop_62_m_dense_23_kernel:@@7
%assignvariableop_63_v_dense_23_kernel:@@1
#assignvariableop_64_m_dense_23_bias:@1
#assignvariableop_65_v_dense_23_bias:@@
2assignvariableop_66_m_batch_normalization_19_gamma:@@
2assignvariableop_67_v_batch_normalization_19_gamma:@?
1assignvariableop_68_m_batch_normalization_19_beta:@?
1assignvariableop_69_v_batch_normalization_19_beta:@4
&assignvariableop_70_m_p_re_lu_19_alpha:@4
&assignvariableop_71_v_p_re_lu_19_alpha:@7
%assignvariableop_72_m_dense_24_kernel:@7
%assignvariableop_73_v_dense_24_kernel:@1
#assignvariableop_74_m_dense_24_bias:1
#assignvariableop_75_v_dense_24_bias:%
assignvariableop_76_total_1: %
assignvariableop_77_count_1: #
assignvariableop_78_total: #
assignvariableop_79_count: 
identity_81��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_9�"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*�"
value�"B�"QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*�
value�B�QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*_
dtypesU
S2Q	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_16_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_16_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_16_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_16_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_p_re_lu_16_alphaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_21_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_21_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_17_gammaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_17_betaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp6assignvariableop_11_batch_normalization_17_moving_meanIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp:assignvariableop_12_batch_normalization_17_moving_varianceIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_p_re_lu_17_alphaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_22_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_22_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_18_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_18_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_18_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_18_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp$assignvariableop_20_p_re_lu_18_alphaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_23_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp!assignvariableop_22_dense_23_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp0assignvariableop_23_batch_normalization_19_gammaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_19_betaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_batch_normalization_19_moving_meanIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp:assignvariableop_26_batch_normalization_19_moving_varianceIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp$assignvariableop_27_p_re_lu_19_alphaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_24_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_dense_24_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_iterationIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_current_learning_rateIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp%assignvariableop_32_m_dense_20_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp%assignvariableop_33_v_dense_20_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_m_dense_20_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp#assignvariableop_35_v_dense_20_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp2assignvariableop_36_m_batch_normalization_16_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp2assignvariableop_37_v_batch_normalization_16_gammaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp1assignvariableop_38_m_batch_normalization_16_betaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp1assignvariableop_39_v_batch_normalization_16_betaIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp&assignvariableop_40_m_p_re_lu_16_alphaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp&assignvariableop_41_v_p_re_lu_16_alphaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp%assignvariableop_42_m_dense_21_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp%assignvariableop_43_v_dense_21_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp#assignvariableop_44_m_dense_21_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp#assignvariableop_45_v_dense_21_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp2assignvariableop_46_m_batch_normalization_17_gammaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp2assignvariableop_47_v_batch_normalization_17_gammaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp1assignvariableop_48_m_batch_normalization_17_betaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp1assignvariableop_49_v_batch_normalization_17_betaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp&assignvariableop_50_m_p_re_lu_17_alphaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp&assignvariableop_51_v_p_re_lu_17_alphaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp%assignvariableop_52_m_dense_22_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp%assignvariableop_53_v_dense_22_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp#assignvariableop_54_m_dense_22_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp#assignvariableop_55_v_dense_22_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp2assignvariableop_56_m_batch_normalization_18_gammaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp2assignvariableop_57_v_batch_normalization_18_gammaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp1assignvariableop_58_m_batch_normalization_18_betaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp1assignvariableop_59_v_batch_normalization_18_betaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp&assignvariableop_60_m_p_re_lu_18_alphaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp&assignvariableop_61_v_p_re_lu_18_alphaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp%assignvariableop_62_m_dense_23_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp%assignvariableop_63_v_dense_23_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp#assignvariableop_64_m_dense_23_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp#assignvariableop_65_v_dense_23_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp2assignvariableop_66_m_batch_normalization_19_gammaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp2assignvariableop_67_v_batch_normalization_19_gammaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp1assignvariableop_68_m_batch_normalization_19_betaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp1assignvariableop_69_v_batch_normalization_19_betaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp&assignvariableop_70_m_p_re_lu_19_alphaIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp&assignvariableop_71_v_p_re_lu_19_alphaIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp%assignvariableop_72_m_dense_24_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp%assignvariableop_73_v_dense_24_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp#assignvariableop_74_m_dense_24_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp#assignvariableop_75_v_dense_24_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOpassignvariableop_76_total_1Identity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOpassignvariableop_77_count_1Identity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpassignvariableop_78_totalIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOpassignvariableop_79_countIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_80Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_81IdentityIdentity_80:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_81Identity_81:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%P!

_user_specified_namecount:%O!

_user_specified_nametotal:'N#
!
_user_specified_name	count_1:'M#
!
_user_specified_name	total_1:/L+
)
_user_specified_namev/dense_24/bias:/K+
)
_user_specified_namem/dense_24/bias:1J-
+
_user_specified_namev/dense_24/kernel:1I-
+
_user_specified_namem/dense_24/kernel:2H.
,
_user_specified_namev/p_re_lu_19/alpha:2G.
,
_user_specified_namem/p_re_lu_19/alpha:=F9
7
_user_specified_namev/batch_normalization_19/beta:=E9
7
_user_specified_namem/batch_normalization_19/beta:>D:
8
_user_specified_name v/batch_normalization_19/gamma:>C:
8
_user_specified_name m/batch_normalization_19/gamma:/B+
)
_user_specified_namev/dense_23/bias:/A+
)
_user_specified_namem/dense_23/bias:1@-
+
_user_specified_namev/dense_23/kernel:1?-
+
_user_specified_namem/dense_23/kernel:2>.
,
_user_specified_namev/p_re_lu_18/alpha:2=.
,
_user_specified_namem/p_re_lu_18/alpha:=<9
7
_user_specified_namev/batch_normalization_18/beta:=;9
7
_user_specified_namem/batch_normalization_18/beta:>::
8
_user_specified_name v/batch_normalization_18/gamma:>9:
8
_user_specified_name m/batch_normalization_18/gamma:/8+
)
_user_specified_namev/dense_22/bias:/7+
)
_user_specified_namem/dense_22/bias:16-
+
_user_specified_namev/dense_22/kernel:15-
+
_user_specified_namem/dense_22/kernel:24.
,
_user_specified_namev/p_re_lu_17/alpha:23.
,
_user_specified_namem/p_re_lu_17/alpha:=29
7
_user_specified_namev/batch_normalization_17/beta:=19
7
_user_specified_namem/batch_normalization_17/beta:>0:
8
_user_specified_name v/batch_normalization_17/gamma:>/:
8
_user_specified_name m/batch_normalization_17/gamma:/.+
)
_user_specified_namev/dense_21/bias:/-+
)
_user_specified_namem/dense_21/bias:1,-
+
_user_specified_namev/dense_21/kernel:1+-
+
_user_specified_namem/dense_21/kernel:2*.
,
_user_specified_namev/p_re_lu_16/alpha:2).
,
_user_specified_namem/p_re_lu_16/alpha:=(9
7
_user_specified_namev/batch_normalization_16/beta:='9
7
_user_specified_namem/batch_normalization_16/beta:>&:
8
_user_specified_name v/batch_normalization_16/gamma:>%:
8
_user_specified_name m/batch_normalization_16/gamma:/$+
)
_user_specified_namev/dense_20/bias:/#+
)
_user_specified_namem/dense_20/bias:1"-
+
_user_specified_namev/dense_20/kernel:1!-
+
_user_specified_namem/dense_20/kernel:5 1
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_24/bias:/+
)
_user_specified_namedense_24/kernel:0,
*
_user_specified_namep_re_lu_19/alpha:FB
@
_user_specified_name(&batch_normalization_19/moving_variance:B>
<
_user_specified_name$"batch_normalization_19/moving_mean:;7
5
_user_specified_namebatch_normalization_19/beta:<8
6
_user_specified_namebatch_normalization_19/gamma:-)
'
_user_specified_namedense_23/bias:/+
)
_user_specified_namedense_23/kernel:0,
*
_user_specified_namep_re_lu_18/alpha:FB
@
_user_specified_name(&batch_normalization_18/moving_variance:B>
<
_user_specified_name$"batch_normalization_18/moving_mean:;7
5
_user_specified_namebatch_normalization_18/beta:<8
6
_user_specified_namebatch_normalization_18/gamma:-)
'
_user_specified_namedense_22/bias:/+
)
_user_specified_namedense_22/kernel:0,
*
_user_specified_namep_re_lu_17/alpha:FB
@
_user_specified_name(&batch_normalization_17/moving_variance:B>
<
_user_specified_name$"batch_normalization_17/moving_mean:;7
5
_user_specified_namebatch_normalization_17/beta:<
8
6
_user_specified_namebatch_normalization_17/gamma:-	)
'
_user_specified_namedense_21/bias:/+
)
_user_specified_namedense_21/kernel:0,
*
_user_specified_namep_re_lu_16/alpha:FB
@
_user_specified_name(&batch_normalization_16/moving_variance:B>
<
_user_specified_name$"batch_normalization_16/moving_mean:;7
5
_user_specified_namebatch_normalization_16/beta:<8
6
_user_specified_namebatch_normalization_16/gamma:-)
'
_user_specified_namedense_20/bias:/+
)
_user_specified_namedense_20/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�S
�
G__inference_sequential_4_layer_call_and_return_conditional_losses_94962
dense_20_input 
dense_20_94888:@
dense_20_94890:@*
batch_normalization_16_94893:@*
batch_normalization_16_94895:@*
batch_normalization_16_94897:@*
batch_normalization_16_94899:@
p_re_lu_16_94902:@ 
dense_21_94905:@@
dense_21_94907:@*
batch_normalization_17_94910:@*
batch_normalization_17_94912:@*
batch_normalization_17_94914:@*
batch_normalization_17_94916:@
p_re_lu_17_94919:@ 
dense_22_94922:@@
dense_22_94924:@*
batch_normalization_18_94927:@*
batch_normalization_18_94929:@*
batch_normalization_18_94931:@*
batch_normalization_18_94933:@
p_re_lu_18_94936:@ 
dense_23_94939:@@
dense_23_94941:@*
batch_normalization_19_94944:@*
batch_normalization_19_94946:@*
batch_normalization_19_94948:@*
batch_normalization_19_94950:@
p_re_lu_19_94953:@ 
dense_24_94956:@
dense_24_94958:
identity��.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�.batch_normalization_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�"p_re_lu_16/StatefulPartitionedCall�"p_re_lu_17/StatefulPartitionedCall�"p_re_lu_18/StatefulPartitionedCall�"p_re_lu_19/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_inputdense_20_94888dense_20_94890*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_94769�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_16_94893batch_normalization_16_94895batch_normalization_16_94897batch_normalization_16_94899*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_94415�
"p_re_lu_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0p_re_lu_16_94902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_16_layer_call_and_return_conditional_losses_94453�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall+p_re_lu_16/StatefulPartitionedCall:output:0dense_21_94905dense_21_94907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_94796�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_17_94910batch_normalization_17_94912batch_normalization_17_94914batch_normalization_17_94916*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_94514�
"p_re_lu_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0p_re_lu_17_94919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_17_layer_call_and_return_conditional_losses_94552�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+p_re_lu_17/StatefulPartitionedCall:output:0dense_22_94922dense_22_94924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_94823�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_18_94927batch_normalization_18_94929batch_normalization_18_94931batch_normalization_18_94933*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_94613�
"p_re_lu_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0p_re_lu_18_94936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_18_layer_call_and_return_conditional_losses_94651�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+p_re_lu_18/StatefulPartitionedCall:output:0dense_23_94939dense_23_94941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_94850�
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_19_94944batch_normalization_19_94946batch_normalization_19_94948batch_normalization_19_94950*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_94712�
"p_re_lu_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0p_re_lu_19_94953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_19_layer_call_and_return_conditional_losses_94750�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+p_re_lu_19/StatefulPartitionedCall:output:0dense_24_94956dense_24_94958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_94878x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^p_re_lu_16/StatefulPartitionedCall#^p_re_lu_17/StatefulPartitionedCall#^p_re_lu_18/StatefulPartitionedCall#^p_re_lu_19/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"p_re_lu_16/StatefulPartitionedCall"p_re_lu_16/StatefulPartitionedCall2H
"p_re_lu_17/StatefulPartitionedCall"p_re_lu_17/StatefulPartitionedCall2H
"p_re_lu_18/StatefulPartitionedCall"p_re_lu_18/StatefulPartitionedCall2H
"p_re_lu_19/StatefulPartitionedCall"p_re_lu_19/StatefulPartitionedCall:%!

_user_specified_name94958:%!

_user_specified_name94956:%!

_user_specified_name94953:%!

_user_specified_name94950:%!

_user_specified_name94948:%!

_user_specified_name94946:%!

_user_specified_name94944:%!

_user_specified_name94941:%!

_user_specified_name94939:%!

_user_specified_name94936:%!

_user_specified_name94933:%!

_user_specified_name94931:%!

_user_specified_name94929:%!

_user_specified_name94927:%!

_user_specified_name94924:%!

_user_specified_name94922:%!

_user_specified_name94919:%!

_user_specified_name94916:%!

_user_specified_name94914:%!

_user_specified_name94912:%
!

_user_specified_name94910:%	!

_user_specified_name94907:%!

_user_specified_name94905:%!

_user_specified_name94902:%!

_user_specified_name94899:%!

_user_specified_name94897:%!

_user_specified_name94895:%!

_user_specified_name94893:%!

_user_specified_name94890:%!

_user_specified_name94888:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�
�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_94514

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_95656

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_17_layer_call_fn_95366

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_94514o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95362:%!

_user_specified_name95360:%!

_user_specified_name95358:%!

_user_specified_name95356:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_dense_20_layer_call_and_return_conditional_losses_94769

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_95518

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
z
*__inference_p_re_lu_17_layer_call_fn_95427

inputs
unknown:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_17_layer_call_and_return_conditional_losses_94552o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95423:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
E__inference_p_re_lu_17_layer_call_and_return_conditional_losses_95439

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_94613

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_sequential_4_layer_call_fn_95092
dense_20_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_94962o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95088:%!

_user_specified_name95086:%!

_user_specified_name95084:%!

_user_specified_name95082:%!

_user_specified_name95080:%!

_user_specified_name95078:%!

_user_specified_name95076:%!

_user_specified_name95074:%!

_user_specified_name95072:%!

_user_specified_name95070:%!

_user_specified_name95068:%!

_user_specified_name95066:%!

_user_specified_name95064:%!

_user_specified_name95062:%!

_user_specified_name95060:%!

_user_specified_name95058:%!

_user_specified_name95056:%!

_user_specified_name95054:%!

_user_specified_name95052:%!

_user_specified_name95050:%
!

_user_specified_name95048:%	!

_user_specified_name95046:%!

_user_specified_name95044:%!

_user_specified_name95042:%!

_user_specified_name95040:%!

_user_specified_name95038:%!

_user_specified_name95036:%!

_user_specified_name95034:%!

_user_specified_name95032:%!

_user_specified_name95030:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�	
�
E__inference_p_re_lu_16_layer_call_and_return_conditional_losses_94453

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
(__inference_dense_20_layer_call_fn_95212

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_94769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95208:%!

_user_specified_name95206:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_p_re_lu_17_layer_call_and_return_conditional_losses_94552

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
z
*__inference_p_re_lu_19_layer_call_fn_95663

inputs
unknown:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_p_re_lu_19_layer_call_and_return_conditional_losses_94750o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95659:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
(__inference_dense_23_layer_call_fn_95566

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_94850o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95562:%!

_user_specified_name95560:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_16_layer_call_fn_95235

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_94395o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95231:%!

_user_specified_name95229:%!

_user_specified_name95227:%!

_user_specified_name95225:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�&
�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_95282

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_16_layer_call_fn_95248

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_94415o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95244:%!

_user_specified_name95242:%!

_user_specified_name95240:%!

_user_specified_name95238:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_94415

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_94712

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_sequential_4_layer_call_fn_95027
dense_20_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_94885o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95023:%!

_user_specified_name95021:%!

_user_specified_name95019:%!

_user_specified_name95017:%!

_user_specified_name95015:%!

_user_specified_name95013:%!

_user_specified_name95011:%!

_user_specified_name95009:%!

_user_specified_name95007:%!

_user_specified_name95005:%!

_user_specified_name95003:%!

_user_specified_name95001:%!

_user_specified_name94999:%!

_user_specified_name94997:%!

_user_specified_name94995:%!

_user_specified_name94993:%!

_user_specified_name94991:%!

_user_specified_name94989:%!

_user_specified_name94987:%!

_user_specified_name94985:%
!

_user_specified_name94983:%	!

_user_specified_name94981:%!

_user_specified_name94979:%!

_user_specified_name94977:%!

_user_specified_name94975:%!

_user_specified_name94973:%!

_user_specified_name94971:%!

_user_specified_name94969:%!

_user_specified_name94967:%!

_user_specified_name94965:W S
'
_output_shapes
:���������
(
_user_specified_namedense_20_input
�	
�
E__inference_p_re_lu_19_layer_call_and_return_conditional_losses_95675

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_18_layer_call_fn_95484

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_94613o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95480:%!

_user_specified_name95478:%!

_user_specified_name95476:%!

_user_specified_name95474:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�&
�
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_94692

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
E__inference_p_re_lu_18_layer_call_and_return_conditional_losses_95557

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
(__inference_dense_21_layer_call_fn_95330

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_94796o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95326:%!

_user_specified_name95324:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�&
�
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_95636

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_18_layer_call_fn_95471

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_94593o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name95467:%!

_user_specified_name95465:%!

_user_specified_name95463:%!

_user_specified_name95461:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�&
�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_94494

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
E__inference_p_re_lu_18_layer_call_and_return_conditional_losses_94651

inputs%
readvariableop_resource:@
identity��ReadVariableOpO
ReluReluinputs*
T0*0
_output_shapes
:������������������b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0G
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@O
Neg_1Neginputs*
T0*0
_output_shapes
:������������������T
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������[
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@[
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:������������������: 2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�&
�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_94395

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�I
__inference__traced_save_96197
file_prefix8
&read_disablecopyonread_dense_20_kernel:@4
&read_1_disablecopyonread_dense_20_bias:@C
5read_2_disablecopyonread_batch_normalization_16_gamma:@B
4read_3_disablecopyonread_batch_normalization_16_beta:@I
;read_4_disablecopyonread_batch_normalization_16_moving_mean:@M
?read_5_disablecopyonread_batch_normalization_16_moving_variance:@7
)read_6_disablecopyonread_p_re_lu_16_alpha:@:
(read_7_disablecopyonread_dense_21_kernel:@@4
&read_8_disablecopyonread_dense_21_bias:@C
5read_9_disablecopyonread_batch_normalization_17_gamma:@C
5read_10_disablecopyonread_batch_normalization_17_beta:@J
<read_11_disablecopyonread_batch_normalization_17_moving_mean:@N
@read_12_disablecopyonread_batch_normalization_17_moving_variance:@8
*read_13_disablecopyonread_p_re_lu_17_alpha:@;
)read_14_disablecopyonread_dense_22_kernel:@@5
'read_15_disablecopyonread_dense_22_bias:@D
6read_16_disablecopyonread_batch_normalization_18_gamma:@C
5read_17_disablecopyonread_batch_normalization_18_beta:@J
<read_18_disablecopyonread_batch_normalization_18_moving_mean:@N
@read_19_disablecopyonread_batch_normalization_18_moving_variance:@8
*read_20_disablecopyonread_p_re_lu_18_alpha:@;
)read_21_disablecopyonread_dense_23_kernel:@@5
'read_22_disablecopyonread_dense_23_bias:@D
6read_23_disablecopyonread_batch_normalization_19_gamma:@C
5read_24_disablecopyonread_batch_normalization_19_beta:@J
<read_25_disablecopyonread_batch_normalization_19_moving_mean:@N
@read_26_disablecopyonread_batch_normalization_19_moving_variance:@8
*read_27_disablecopyonread_p_re_lu_19_alpha:@;
)read_28_disablecopyonread_dense_24_kernel:@5
'read_29_disablecopyonread_dense_24_bias:-
#read_30_disablecopyonread_iteration:	 9
/read_31_disablecopyonread_current_learning_rate: =
+read_32_disablecopyonread_m_dense_20_kernel:@=
+read_33_disablecopyonread_v_dense_20_kernel:@7
)read_34_disablecopyonread_m_dense_20_bias:@7
)read_35_disablecopyonread_v_dense_20_bias:@F
8read_36_disablecopyonread_m_batch_normalization_16_gamma:@F
8read_37_disablecopyonread_v_batch_normalization_16_gamma:@E
7read_38_disablecopyonread_m_batch_normalization_16_beta:@E
7read_39_disablecopyonread_v_batch_normalization_16_beta:@:
,read_40_disablecopyonread_m_p_re_lu_16_alpha:@:
,read_41_disablecopyonread_v_p_re_lu_16_alpha:@=
+read_42_disablecopyonread_m_dense_21_kernel:@@=
+read_43_disablecopyonread_v_dense_21_kernel:@@7
)read_44_disablecopyonread_m_dense_21_bias:@7
)read_45_disablecopyonread_v_dense_21_bias:@F
8read_46_disablecopyonread_m_batch_normalization_17_gamma:@F
8read_47_disablecopyonread_v_batch_normalization_17_gamma:@E
7read_48_disablecopyonread_m_batch_normalization_17_beta:@E
7read_49_disablecopyonread_v_batch_normalization_17_beta:@:
,read_50_disablecopyonread_m_p_re_lu_17_alpha:@:
,read_51_disablecopyonread_v_p_re_lu_17_alpha:@=
+read_52_disablecopyonread_m_dense_22_kernel:@@=
+read_53_disablecopyonread_v_dense_22_kernel:@@7
)read_54_disablecopyonread_m_dense_22_bias:@7
)read_55_disablecopyonread_v_dense_22_bias:@F
8read_56_disablecopyonread_m_batch_normalization_18_gamma:@F
8read_57_disablecopyonread_v_batch_normalization_18_gamma:@E
7read_58_disablecopyonread_m_batch_normalization_18_beta:@E
7read_59_disablecopyonread_v_batch_normalization_18_beta:@:
,read_60_disablecopyonread_m_p_re_lu_18_alpha:@:
,read_61_disablecopyonread_v_p_re_lu_18_alpha:@=
+read_62_disablecopyonread_m_dense_23_kernel:@@=
+read_63_disablecopyonread_v_dense_23_kernel:@@7
)read_64_disablecopyonread_m_dense_23_bias:@7
)read_65_disablecopyonread_v_dense_23_bias:@F
8read_66_disablecopyonread_m_batch_normalization_19_gamma:@F
8read_67_disablecopyonread_v_batch_normalization_19_gamma:@E
7read_68_disablecopyonread_m_batch_normalization_19_beta:@E
7read_69_disablecopyonread_v_batch_normalization_19_beta:@:
,read_70_disablecopyonread_m_p_re_lu_19_alpha:@:
,read_71_disablecopyonread_v_p_re_lu_19_alpha:@=
+read_72_disablecopyonread_m_dense_24_kernel:@=
+read_73_disablecopyonread_v_dense_24_kernel:@7
)read_74_disablecopyonread_m_dense_24_bias:7
)read_75_disablecopyonread_v_dense_24_bias:+
!read_76_disablecopyonread_total_1: +
!read_77_disablecopyonread_count_1: )
read_78_disablecopyonread_total: )
read_79_disablecopyonread_count: 
savev2_const
identity_161��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_20_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:@z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_20_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_2/DisableCopyOnReadDisableCopyOnRead5read_2_disablecopyonread_batch_normalization_16_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp5read_2_disablecopyonread_batch_normalization_16_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_3/DisableCopyOnReadDisableCopyOnRead4read_3_disablecopyonread_batch_normalization_16_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp4read_3_disablecopyonread_batch_normalization_16_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_4/DisableCopyOnReadDisableCopyOnRead;read_4_disablecopyonread_batch_normalization_16_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp;read_4_disablecopyonread_batch_normalization_16_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_5/DisableCopyOnReadDisableCopyOnRead?read_5_disablecopyonread_batch_normalization_16_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp?read_5_disablecopyonread_batch_normalization_16_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_p_re_lu_16_alpha"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_p_re_lu_16_alpha^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_dense_21_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:@@z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_dense_21_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_9/DisableCopyOnReadDisableCopyOnRead5read_9_disablecopyonread_batch_normalization_17_gamma"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp5read_9_disablecopyonread_batch_normalization_17_gamma^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_10/DisableCopyOnReadDisableCopyOnRead5read_10_disablecopyonread_batch_normalization_17_beta"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp5read_10_disablecopyonread_batch_normalization_17_beta^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_11/DisableCopyOnReadDisableCopyOnRead<read_11_disablecopyonread_batch_normalization_17_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp<read_11_disablecopyonread_batch_normalization_17_moving_mean^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_12/DisableCopyOnReadDisableCopyOnRead@read_12_disablecopyonread_batch_normalization_17_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp@read_12_disablecopyonread_batch_normalization_17_moving_variance^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_13/DisableCopyOnReadDisableCopyOnRead*read_13_disablecopyonread_p_re_lu_17_alpha"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp*read_13_disablecopyonread_p_re_lu_17_alpha^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_dense_22_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@@|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_dense_22_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_16/DisableCopyOnReadDisableCopyOnRead6read_16_disablecopyonread_batch_normalization_18_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp6read_16_disablecopyonread_batch_normalization_18_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_17/DisableCopyOnReadDisableCopyOnRead5read_17_disablecopyonread_batch_normalization_18_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp5read_17_disablecopyonread_batch_normalization_18_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_18/DisableCopyOnReadDisableCopyOnRead<read_18_disablecopyonread_batch_normalization_18_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp<read_18_disablecopyonread_batch_normalization_18_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_19/DisableCopyOnReadDisableCopyOnRead@read_19_disablecopyonread_batch_normalization_18_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp@read_19_disablecopyonread_batch_normalization_18_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_20/DisableCopyOnReadDisableCopyOnRead*read_20_disablecopyonread_p_re_lu_18_alpha"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp*read_20_disablecopyonread_p_re_lu_18_alpha^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_21/DisableCopyOnReadDisableCopyOnRead)read_21_disablecopyonread_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp)read_21_disablecopyonread_dense_23_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:@@|
Read_22/DisableCopyOnReadDisableCopyOnRead'read_22_disablecopyonread_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp'read_22_disablecopyonread_dense_23_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_23/DisableCopyOnReadDisableCopyOnRead6read_23_disablecopyonread_batch_normalization_19_gamma"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp6read_23_disablecopyonread_batch_normalization_19_gamma^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_24/DisableCopyOnReadDisableCopyOnRead5read_24_disablecopyonread_batch_normalization_19_beta"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp5read_24_disablecopyonread_batch_normalization_19_beta^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_25/DisableCopyOnReadDisableCopyOnRead<read_25_disablecopyonread_batch_normalization_19_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp<read_25_disablecopyonread_batch_normalization_19_moving_mean^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_26/DisableCopyOnReadDisableCopyOnRead@read_26_disablecopyonread_batch_normalization_19_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp@read_26_disablecopyonread_batch_normalization_19_moving_variance^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_27/DisableCopyOnReadDisableCopyOnRead*read_27_disablecopyonread_p_re_lu_19_alpha"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp*read_27_disablecopyonread_p_re_lu_19_alpha^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_28/DisableCopyOnReadDisableCopyOnRead)read_28_disablecopyonread_dense_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp)read_28_disablecopyonread_dense_24_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes

:@|
Read_29/DisableCopyOnReadDisableCopyOnRead'read_29_disablecopyonread_dense_24_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp'read_29_disablecopyonread_dense_24_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_30/DisableCopyOnReadDisableCopyOnRead#read_30_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp#read_30_disablecopyonread_iteration^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0	*
_output_shapes
: �
Read_31/DisableCopyOnReadDisableCopyOnRead/read_31_disablecopyonread_current_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp/read_31_disablecopyonread_current_learning_rate^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_32/DisableCopyOnReadDisableCopyOnRead+read_32_disablecopyonread_m_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp+read_32_disablecopyonread_m_dense_20_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_33/DisableCopyOnReadDisableCopyOnRead+read_33_disablecopyonread_v_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp+read_33_disablecopyonread_v_dense_20_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:@~
Read_34/DisableCopyOnReadDisableCopyOnRead)read_34_disablecopyonread_m_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp)read_34_disablecopyonread_m_dense_20_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_35/DisableCopyOnReadDisableCopyOnRead)read_35_disablecopyonread_v_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp)read_35_disablecopyonread_v_dense_20_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_36/DisableCopyOnReadDisableCopyOnRead8read_36_disablecopyonread_m_batch_normalization_16_gamma"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp8read_36_disablecopyonread_m_batch_normalization_16_gamma^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_37/DisableCopyOnReadDisableCopyOnRead8read_37_disablecopyonread_v_batch_normalization_16_gamma"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp8read_37_disablecopyonread_v_batch_normalization_16_gamma^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_38/DisableCopyOnReadDisableCopyOnRead7read_38_disablecopyonread_m_batch_normalization_16_beta"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp7read_38_disablecopyonread_m_batch_normalization_16_beta^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_39/DisableCopyOnReadDisableCopyOnRead7read_39_disablecopyonread_v_batch_normalization_16_beta"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp7read_39_disablecopyonread_v_batch_normalization_16_beta^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_40/DisableCopyOnReadDisableCopyOnRead,read_40_disablecopyonread_m_p_re_lu_16_alpha"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp,read_40_disablecopyonread_m_p_re_lu_16_alpha^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_41/DisableCopyOnReadDisableCopyOnRead,read_41_disablecopyonread_v_p_re_lu_16_alpha"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp,read_41_disablecopyonread_v_p_re_lu_16_alpha^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_42/DisableCopyOnReadDisableCopyOnRead+read_42_disablecopyonread_m_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp+read_42_disablecopyonread_m_dense_21_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_43/DisableCopyOnReadDisableCopyOnRead+read_43_disablecopyonread_v_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp+read_43_disablecopyonread_v_dense_21_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:@@~
Read_44/DisableCopyOnReadDisableCopyOnRead)read_44_disablecopyonread_m_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp)read_44_disablecopyonread_m_dense_21_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_45/DisableCopyOnReadDisableCopyOnRead)read_45_disablecopyonread_v_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp)read_45_disablecopyonread_v_dense_21_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_46/DisableCopyOnReadDisableCopyOnRead8read_46_disablecopyonread_m_batch_normalization_17_gamma"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp8read_46_disablecopyonread_m_batch_normalization_17_gamma^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_47/DisableCopyOnReadDisableCopyOnRead8read_47_disablecopyonread_v_batch_normalization_17_gamma"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp8read_47_disablecopyonread_v_batch_normalization_17_gamma^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_48/DisableCopyOnReadDisableCopyOnRead7read_48_disablecopyonread_m_batch_normalization_17_beta"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp7read_48_disablecopyonread_m_batch_normalization_17_beta^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_49/DisableCopyOnReadDisableCopyOnRead7read_49_disablecopyonread_v_batch_normalization_17_beta"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp7read_49_disablecopyonread_v_batch_normalization_17_beta^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_50/DisableCopyOnReadDisableCopyOnRead,read_50_disablecopyonread_m_p_re_lu_17_alpha"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp,read_50_disablecopyonread_m_p_re_lu_17_alpha^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_51/DisableCopyOnReadDisableCopyOnRead,read_51_disablecopyonread_v_p_re_lu_17_alpha"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp,read_51_disablecopyonread_v_p_re_lu_17_alpha^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_52/DisableCopyOnReadDisableCopyOnRead+read_52_disablecopyonread_m_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp+read_52_disablecopyonread_m_dense_22_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_53/DisableCopyOnReadDisableCopyOnRead+read_53_disablecopyonread_v_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp+read_53_disablecopyonread_v_dense_22_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes

:@@~
Read_54/DisableCopyOnReadDisableCopyOnRead)read_54_disablecopyonread_m_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp)read_54_disablecopyonread_m_dense_22_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_55/DisableCopyOnReadDisableCopyOnRead)read_55_disablecopyonread_v_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp)read_55_disablecopyonread_v_dense_22_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_56/DisableCopyOnReadDisableCopyOnRead8read_56_disablecopyonread_m_batch_normalization_18_gamma"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp8read_56_disablecopyonread_m_batch_normalization_18_gamma^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_57/DisableCopyOnReadDisableCopyOnRead8read_57_disablecopyonread_v_batch_normalization_18_gamma"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp8read_57_disablecopyonread_v_batch_normalization_18_gamma^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_58/DisableCopyOnReadDisableCopyOnRead7read_58_disablecopyonread_m_batch_normalization_18_beta"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp7read_58_disablecopyonread_m_batch_normalization_18_beta^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_59/DisableCopyOnReadDisableCopyOnRead7read_59_disablecopyonread_v_batch_normalization_18_beta"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp7read_59_disablecopyonread_v_batch_normalization_18_beta^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_60/DisableCopyOnReadDisableCopyOnRead,read_60_disablecopyonread_m_p_re_lu_18_alpha"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp,read_60_disablecopyonread_m_p_re_lu_18_alpha^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_61/DisableCopyOnReadDisableCopyOnRead,read_61_disablecopyonread_v_p_re_lu_18_alpha"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp,read_61_disablecopyonread_v_p_re_lu_18_alpha^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_62/DisableCopyOnReadDisableCopyOnRead+read_62_disablecopyonread_m_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp+read_62_disablecopyonread_m_dense_23_kernel^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_63/DisableCopyOnReadDisableCopyOnRead+read_63_disablecopyonread_v_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp+read_63_disablecopyonread_v_dense_23_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes

:@@~
Read_64/DisableCopyOnReadDisableCopyOnRead)read_64_disablecopyonread_m_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp)read_64_disablecopyonread_m_dense_23_bias^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_65/DisableCopyOnReadDisableCopyOnRead)read_65_disablecopyonread_v_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp)read_65_disablecopyonread_v_dense_23_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_66/DisableCopyOnReadDisableCopyOnRead8read_66_disablecopyonread_m_batch_normalization_19_gamma"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp8read_66_disablecopyonread_m_batch_normalization_19_gamma^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_67/DisableCopyOnReadDisableCopyOnRead8read_67_disablecopyonread_v_batch_normalization_19_gamma"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp8read_67_disablecopyonread_v_batch_normalization_19_gamma^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_68/DisableCopyOnReadDisableCopyOnRead7read_68_disablecopyonread_m_batch_normalization_19_beta"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp7read_68_disablecopyonread_m_batch_normalization_19_beta^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_69/DisableCopyOnReadDisableCopyOnRead7read_69_disablecopyonread_v_batch_normalization_19_beta"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp7read_69_disablecopyonread_v_batch_normalization_19_beta^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_70/DisableCopyOnReadDisableCopyOnRead,read_70_disablecopyonread_m_p_re_lu_19_alpha"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp,read_70_disablecopyonread_m_p_re_lu_19_alpha^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_71/DisableCopyOnReadDisableCopyOnRead,read_71_disablecopyonread_v_p_re_lu_19_alpha"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp,read_71_disablecopyonread_v_p_re_lu_19_alpha^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_72/DisableCopyOnReadDisableCopyOnRead+read_72_disablecopyonread_m_dense_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp+read_72_disablecopyonread_m_dense_24_kernel^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_73/DisableCopyOnReadDisableCopyOnRead+read_73_disablecopyonread_v_dense_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp+read_73_disablecopyonread_v_dense_24_kernel^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes

:@~
Read_74/DisableCopyOnReadDisableCopyOnRead)read_74_disablecopyonread_m_dense_24_bias"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp)read_74_disablecopyonread_m_dense_24_bias^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_75/DisableCopyOnReadDisableCopyOnRead)read_75_disablecopyonread_v_dense_24_bias"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp)read_75_disablecopyonread_v_dense_24_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_76/DisableCopyOnReadDisableCopyOnRead!read_76_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp!read_76_disablecopyonread_total_1^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_77/DisableCopyOnReadDisableCopyOnRead!read_77_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp!read_77_disablecopyonread_count_1^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_78/DisableCopyOnReadDisableCopyOnReadread_78_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOpread_78_disablecopyonread_total^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_79/DisableCopyOnReadDisableCopyOnReadread_79_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOpread_79_disablecopyonread_count^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
: �"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*�"
value�"B�"QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*�
value�B�QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *_
dtypesU
S2Q	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_160Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_161IdentityIdentity_160:output:0^NoOp*
T0*
_output_shapes
: �!
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_161Identity_161:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=Q9

_output_shapes
: 

_user_specified_nameConst:%P!

_user_specified_namecount:%O!

_user_specified_nametotal:'N#
!
_user_specified_name	count_1:'M#
!
_user_specified_name	total_1:/L+
)
_user_specified_namev/dense_24/bias:/K+
)
_user_specified_namem/dense_24/bias:1J-
+
_user_specified_namev/dense_24/kernel:1I-
+
_user_specified_namem/dense_24/kernel:2H.
,
_user_specified_namev/p_re_lu_19/alpha:2G.
,
_user_specified_namem/p_re_lu_19/alpha:=F9
7
_user_specified_namev/batch_normalization_19/beta:=E9
7
_user_specified_namem/batch_normalization_19/beta:>D:
8
_user_specified_name v/batch_normalization_19/gamma:>C:
8
_user_specified_name m/batch_normalization_19/gamma:/B+
)
_user_specified_namev/dense_23/bias:/A+
)
_user_specified_namem/dense_23/bias:1@-
+
_user_specified_namev/dense_23/kernel:1?-
+
_user_specified_namem/dense_23/kernel:2>.
,
_user_specified_namev/p_re_lu_18/alpha:2=.
,
_user_specified_namem/p_re_lu_18/alpha:=<9
7
_user_specified_namev/batch_normalization_18/beta:=;9
7
_user_specified_namem/batch_normalization_18/beta:>::
8
_user_specified_name v/batch_normalization_18/gamma:>9:
8
_user_specified_name m/batch_normalization_18/gamma:/8+
)
_user_specified_namev/dense_22/bias:/7+
)
_user_specified_namem/dense_22/bias:16-
+
_user_specified_namev/dense_22/kernel:15-
+
_user_specified_namem/dense_22/kernel:24.
,
_user_specified_namev/p_re_lu_17/alpha:23.
,
_user_specified_namem/p_re_lu_17/alpha:=29
7
_user_specified_namev/batch_normalization_17/beta:=19
7
_user_specified_namem/batch_normalization_17/beta:>0:
8
_user_specified_name v/batch_normalization_17/gamma:>/:
8
_user_specified_name m/batch_normalization_17/gamma:/.+
)
_user_specified_namev/dense_21/bias:/-+
)
_user_specified_namem/dense_21/bias:1,-
+
_user_specified_namev/dense_21/kernel:1+-
+
_user_specified_namem/dense_21/kernel:2*.
,
_user_specified_namev/p_re_lu_16/alpha:2).
,
_user_specified_namem/p_re_lu_16/alpha:=(9
7
_user_specified_namev/batch_normalization_16/beta:='9
7
_user_specified_namem/batch_normalization_16/beta:>&:
8
_user_specified_name v/batch_normalization_16/gamma:>%:
8
_user_specified_name m/batch_normalization_16/gamma:/$+
)
_user_specified_namev/dense_20/bias:/#+
)
_user_specified_namem/dense_20/bias:1"-
+
_user_specified_namev/dense_20/kernel:1!-
+
_user_specified_namem/dense_20/kernel:5 1
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_24/bias:/+
)
_user_specified_namedense_24/kernel:0,
*
_user_specified_namep_re_lu_19/alpha:FB
@
_user_specified_name(&batch_normalization_19/moving_variance:B>
<
_user_specified_name$"batch_normalization_19/moving_mean:;7
5
_user_specified_namebatch_normalization_19/beta:<8
6
_user_specified_namebatch_normalization_19/gamma:-)
'
_user_specified_namedense_23/bias:/+
)
_user_specified_namedense_23/kernel:0,
*
_user_specified_namep_re_lu_18/alpha:FB
@
_user_specified_name(&batch_normalization_18/moving_variance:B>
<
_user_specified_name$"batch_normalization_18/moving_mean:;7
5
_user_specified_namebatch_normalization_18/beta:<8
6
_user_specified_namebatch_normalization_18/gamma:-)
'
_user_specified_namedense_22/bias:/+
)
_user_specified_namedense_22/kernel:0,
*
_user_specified_namep_re_lu_17/alpha:FB
@
_user_specified_name(&batch_normalization_17/moving_variance:B>
<
_user_specified_name$"batch_normalization_17/moving_mean:;7
5
_user_specified_namebatch_normalization_17/beta:<
8
6
_user_specified_namebatch_normalization_17/gamma:-	)
'
_user_specified_namedense_21/bias:/+
)
_user_specified_namedense_21/kernel:0,
*
_user_specified_namep_re_lu_16/alpha:FB
@
_user_specified_name(&batch_normalization_16/moving_variance:B>
<
_user_specified_name$"batch_normalization_16/moving_mean:;7
5
_user_specified_namebatch_normalization_16/beta:<8
6
_user_specified_namebatch_normalization_16/gamma:-)
'
_user_specified_namedense_20/bias:/+
)
_user_specified_namedense_20/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
C__inference_dense_23_layer_call_and_return_conditional_losses_95576

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
dense_20_input7
 serving_default_dense_20_input:0���������<
dense_240
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
layer_with_weights-11
layer-11
layer_with_weights-12
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%axis
	&gamma
'beta
(moving_mean
)moving_variance"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
	0alpha"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
	Jalpha"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
	dalpha"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
	~alpha"
_tf_keras_layer
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
0
1
&2
'3
(4
)5
06
77
88
@9
A10
B11
C12
J13
Q14
R15
Z16
[17
\18
]19
d20
k21
l22
t23
u24
v25
w26
~27
�28
�29"
trackable_list_wrapper
�
0
1
&2
'3
04
75
86
@7
A8
J9
Q10
R11
Z12
[13
d14
k15
l16
t17
u18
~19
�20
�21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_sequential_4_layer_call_fn_95027
,__inference_sequential_4_layer_call_fn_95092�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_sequential_4_layer_call_and_return_conditional_losses_94885
G__inference_sequential_4_layer_call_and_return_conditional_losses_94962�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
 __inference__wrapped_model_94361dense_20_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_current_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_20_layer_call_fn_95212�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_20_layer_call_and_return_conditional_losses_95222�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:@2dense_20/kernel
:@2dense_20/bias
<
&0
'1
(2
)3"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_16_layer_call_fn_95235
6__inference_batch_normalization_16_layer_call_fn_95248�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_95282
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_95302�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_16/gamma
):'@2batch_normalization_16/beta
2:0@ (2"batch_normalization_16/moving_mean
6:4@ (2&batch_normalization_16/moving_variance
'
00"
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_p_re_lu_16_layer_call_fn_95309�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_p_re_lu_16_layer_call_and_return_conditional_losses_95321�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:@2p_re_lu_16/alpha
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_21_layer_call_fn_95330�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_21_layer_call_and_return_conditional_losses_95340�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:@@2dense_21/kernel
:@2dense_21/bias
<
@0
A1
B2
C3"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_17_layer_call_fn_95353
6__inference_batch_normalization_17_layer_call_fn_95366�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_95400
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_95420�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_17/gamma
):'@2batch_normalization_17/beta
2:0@ (2"batch_normalization_17/moving_mean
6:4@ (2&batch_normalization_17/moving_variance
'
J0"
trackable_list_wrapper
'
J0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_p_re_lu_17_layer_call_fn_95427�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_p_re_lu_17_layer_call_and_return_conditional_losses_95439�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:@2p_re_lu_17/alpha
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_22_layer_call_fn_95448�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_22_layer_call_and_return_conditional_losses_95458�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:@@2dense_22/kernel
:@2dense_22/bias
<
Z0
[1
\2
]3"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_18_layer_call_fn_95471
6__inference_batch_normalization_18_layer_call_fn_95484�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_95518
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_95538�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_18/gamma
):'@2batch_normalization_18/beta
2:0@ (2"batch_normalization_18/moving_mean
6:4@ (2&batch_normalization_18/moving_variance
'
d0"
trackable_list_wrapper
'
d0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_p_re_lu_18_layer_call_fn_95545�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_p_re_lu_18_layer_call_and_return_conditional_losses_95557�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:@2p_re_lu_18/alpha
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_23_layer_call_fn_95566�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_23_layer_call_and_return_conditional_losses_95576�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:@@2dense_23/kernel
:@2dense_23/bias
<
t0
u1
v2
w3"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_19_layer_call_fn_95589
6__inference_batch_normalization_19_layer_call_fn_95602�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_95636
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_95656�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_19/gamma
):'@2batch_normalization_19/beta
2:0@ (2"batch_normalization_19/moving_mean
6:4@ (2&batch_normalization_19/moving_variance
'
~0"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_p_re_lu_19_layer_call_fn_95663�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_p_re_lu_19_layer_call_and_return_conditional_losses_95675�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:@2p_re_lu_19/alpha
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_24_layer_call_fn_95684�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_24_layer_call_and_return_conditional_losses_95695�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:@2dense_24/kernel
:2dense_24/bias
X
(0
)1
B2
C3
\4
]5
v6
w7"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_4_layer_call_fn_95027dense_20_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_4_layer_call_fn_95092dense_20_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_4_layer_call_and_return_conditional_losses_94885dense_20_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_4_layer_call_and_return_conditional_losses_94962dense_20_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44"
trackable_list_wrapper
:	 2	iteration
: 2current_learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
#__inference_signature_wrapper_95203dense_20_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 #

kwonlyargs�
jdense_20_input
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_20_layer_call_fn_95212inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_20_layer_call_and_return_conditional_losses_95222inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_16_layer_call_fn_95235inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_16_layer_call_fn_95248inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_95282inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_95302inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_p_re_lu_16_layer_call_fn_95309inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_p_re_lu_16_layer_call_and_return_conditional_losses_95321inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_21_layer_call_fn_95330inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_21_layer_call_and_return_conditional_losses_95340inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_17_layer_call_fn_95353inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_17_layer_call_fn_95366inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_95400inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_95420inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_p_re_lu_17_layer_call_fn_95427inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_p_re_lu_17_layer_call_and_return_conditional_losses_95439inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_22_layer_call_fn_95448inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_22_layer_call_and_return_conditional_losses_95458inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_18_layer_call_fn_95471inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_18_layer_call_fn_95484inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_95518inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_95538inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_p_re_lu_18_layer_call_fn_95545inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_p_re_lu_18_layer_call_and_return_conditional_losses_95557inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_23_layer_call_fn_95566inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_23_layer_call_and_return_conditional_losses_95576inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_19_layer_call_fn_95589inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_19_layer_call_fn_95602inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_95636inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_95656inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_p_re_lu_19_layer_call_fn_95663inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_p_re_lu_19_layer_call_and_return_conditional_losses_95675inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_24_layer_call_fn_95684inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_24_layer_call_and_return_conditional_losses_95695inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
!:@2m/dense_20/kernel
!:@2v/dense_20/kernel
:@2m/dense_20/bias
:@2v/dense_20/bias
*:(@2m/batch_normalization_16/gamma
*:(@2v/batch_normalization_16/gamma
):'@2m/batch_normalization_16/beta
):'@2v/batch_normalization_16/beta
:@2m/p_re_lu_16/alpha
:@2v/p_re_lu_16/alpha
!:@@2m/dense_21/kernel
!:@@2v/dense_21/kernel
:@2m/dense_21/bias
:@2v/dense_21/bias
*:(@2m/batch_normalization_17/gamma
*:(@2v/batch_normalization_17/gamma
):'@2m/batch_normalization_17/beta
):'@2v/batch_normalization_17/beta
:@2m/p_re_lu_17/alpha
:@2v/p_re_lu_17/alpha
!:@@2m/dense_22/kernel
!:@@2v/dense_22/kernel
:@2m/dense_22/bias
:@2v/dense_22/bias
*:(@2m/batch_normalization_18/gamma
*:(@2v/batch_normalization_18/gamma
):'@2m/batch_normalization_18/beta
):'@2v/batch_normalization_18/beta
:@2m/p_re_lu_18/alpha
:@2v/p_re_lu_18/alpha
!:@@2m/dense_23/kernel
!:@@2v/dense_23/kernel
:@2m/dense_23/bias
:@2v/dense_23/bias
*:(@2m/batch_normalization_19/gamma
*:(@2v/batch_normalization_19/gamma
):'@2m/batch_normalization_19/beta
):'@2v/batch_normalization_19/beta
:@2m/p_re_lu_19/alpha
:@2v/p_re_lu_19/alpha
!:@2m/dense_24/kernel
!:@2v/dense_24/kernel
:2m/dense_24/bias
:2v/dense_24/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
 __inference__wrapped_model_94361� )&('078C@BAJQR]Z\[dklwtvu~��7�4
-�*
(�%
dense_20_input���������
� "3�0
.
dense_24"�
dense_24����������
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_95282m()&'7�4
-�*
 �
inputs���������@
p

 
� ",�)
"�
tensor_0���������@
� �
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_95302m)&('7�4
-�*
 �
inputs���������@
p 

 
� ",�)
"�
tensor_0���������@
� �
6__inference_batch_normalization_16_layer_call_fn_95235b()&'7�4
-�*
 �
inputs���������@
p

 
� "!�
unknown���������@�
6__inference_batch_normalization_16_layer_call_fn_95248b)&('7�4
-�*
 �
inputs���������@
p 

 
� "!�
unknown���������@�
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_95400mBC@A7�4
-�*
 �
inputs���������@
p

 
� ",�)
"�
tensor_0���������@
� �
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_95420mC@BA7�4
-�*
 �
inputs���������@
p 

 
� ",�)
"�
tensor_0���������@
� �
6__inference_batch_normalization_17_layer_call_fn_95353bBC@A7�4
-�*
 �
inputs���������@
p

 
� "!�
unknown���������@�
6__inference_batch_normalization_17_layer_call_fn_95366bC@BA7�4
-�*
 �
inputs���������@
p 

 
� "!�
unknown���������@�
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_95518m\]Z[7�4
-�*
 �
inputs���������@
p

 
� ",�)
"�
tensor_0���������@
� �
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_95538m]Z\[7�4
-�*
 �
inputs���������@
p 

 
� ",�)
"�
tensor_0���������@
� �
6__inference_batch_normalization_18_layer_call_fn_95471b\]Z[7�4
-�*
 �
inputs���������@
p

 
� "!�
unknown���������@�
6__inference_batch_normalization_18_layer_call_fn_95484b]Z\[7�4
-�*
 �
inputs���������@
p 

 
� "!�
unknown���������@�
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_95636mvwtu7�4
-�*
 �
inputs���������@
p

 
� ",�)
"�
tensor_0���������@
� �
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_95656mwtvu7�4
-�*
 �
inputs���������@
p 

 
� ",�)
"�
tensor_0���������@
� �
6__inference_batch_normalization_19_layer_call_fn_95589bvwtu7�4
-�*
 �
inputs���������@
p

 
� "!�
unknown���������@�
6__inference_batch_normalization_19_layer_call_fn_95602bwtvu7�4
-�*
 �
inputs���������@
p 

 
� "!�
unknown���������@�
C__inference_dense_20_layer_call_and_return_conditional_losses_95222c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_20_layer_call_fn_95212X/�,
%�"
 �
inputs���������
� "!�
unknown���������@�
C__inference_dense_21_layer_call_and_return_conditional_losses_95340c78/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_21_layer_call_fn_95330X78/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
C__inference_dense_22_layer_call_and_return_conditional_losses_95458cQR/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_22_layer_call_fn_95448XQR/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
C__inference_dense_23_layer_call_and_return_conditional_losses_95576ckl/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_23_layer_call_fn_95566Xkl/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
C__inference_dense_24_layer_call_and_return_conditional_losses_95695e��/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
(__inference_dense_24_layer_call_fn_95684Z��/�,
%�"
 �
inputs���������@
� "!�
unknown����������
E__inference_p_re_lu_16_layer_call_and_return_conditional_losses_95321k08�5
.�+
)�&
inputs������������������
� ",�)
"�
tensor_0���������@
� �
*__inference_p_re_lu_16_layer_call_fn_95309`08�5
.�+
)�&
inputs������������������
� "!�
unknown���������@�
E__inference_p_re_lu_17_layer_call_and_return_conditional_losses_95439kJ8�5
.�+
)�&
inputs������������������
� ",�)
"�
tensor_0���������@
� �
*__inference_p_re_lu_17_layer_call_fn_95427`J8�5
.�+
)�&
inputs������������������
� "!�
unknown���������@�
E__inference_p_re_lu_18_layer_call_and_return_conditional_losses_95557kd8�5
.�+
)�&
inputs������������������
� ",�)
"�
tensor_0���������@
� �
*__inference_p_re_lu_18_layer_call_fn_95545`d8�5
.�+
)�&
inputs������������������
� "!�
unknown���������@�
E__inference_p_re_lu_19_layer_call_and_return_conditional_losses_95675k~8�5
.�+
)�&
inputs������������������
� ",�)
"�
tensor_0���������@
� �
*__inference_p_re_lu_19_layer_call_fn_95663`~8�5
.�+
)�&
inputs������������������
� "!�
unknown���������@�
G__inference_sequential_4_layer_call_and_return_conditional_losses_94885� ()&'078BC@AJQR\]Z[dklvwtu~��?�<
5�2
(�%
dense_20_input���������
p

 
� ",�)
"�
tensor_0���������
� �
G__inference_sequential_4_layer_call_and_return_conditional_losses_94962� )&('078C@BAJQR]Z\[dklwtvu~��?�<
5�2
(�%
dense_20_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
,__inference_sequential_4_layer_call_fn_95027� ()&'078BC@AJQR\]Z[dklvwtu~��?�<
5�2
(�%
dense_20_input���������
p

 
� "!�
unknown����������
,__inference_sequential_4_layer_call_fn_95092� )&('078C@BAJQR]Z\[dklwtvu~��?�<
5�2
(�%
dense_20_input���������
p 

 
� "!�
unknown����������
#__inference_signature_wrapper_95203� )&('078C@BAJQR]Z\[dklwtvu~��I�F
� 
?�<
:
dense_20_input(�%
dense_20_input���������"3�0
.
dense_24"�
dense_24���������