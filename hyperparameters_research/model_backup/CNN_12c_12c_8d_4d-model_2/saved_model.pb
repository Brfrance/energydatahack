ÞÐ
±
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
list(type)(0
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8Ã§

conv1d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_36/kernel
y
$conv1d_36/kernel/Read/ReadVariableOpReadVariableOpconv1d_36/kernel*"
_output_shapes
:*
dtype0
t
conv1d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_36/bias
m
"conv1d_36/bias/Read/ReadVariableOpReadVariableOpconv1d_36/bias*
_output_shapes
:*
dtype0

batch_normalization_36/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_36/gamma

0batch_normalization_36/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_36/gamma*
_output_shapes
:*
dtype0

batch_normalization_36/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_36/beta

/batch_normalization_36/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_36/beta*
_output_shapes
:*
dtype0

"batch_normalization_36/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_36/moving_mean

6batch_normalization_36/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_36/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_36/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_36/moving_variance

:batch_normalization_36/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_36/moving_variance*
_output_shapes
:*
dtype0

conv1d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_37/kernel
y
$conv1d_37/kernel/Read/ReadVariableOpReadVariableOpconv1d_37/kernel*"
_output_shapes
:*
dtype0
t
conv1d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_37/bias
m
"conv1d_37/bias/Read/ReadVariableOpReadVariableOpconv1d_37/bias*
_output_shapes
:*
dtype0

batch_normalization_37/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_37/gamma

0batch_normalization_37/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_37/gamma*
_output_shapes
:*
dtype0

batch_normalization_37/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_37/beta

/batch_normalization_37/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_37/beta*
_output_shapes
:*
dtype0

"batch_normalization_37/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_37/moving_mean

6batch_normalization_37/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_37/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_37/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_37/moving_variance

:batch_normalization_37/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_37/moving_variance*
_output_shapes
:*
dtype0
{
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ø* 
shared_namedense_54/kernel
t
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel*
_output_shapes
:	Ø*
dtype0
r
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_54/bias
k
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes
:*
dtype0
z
dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_55/kernel
s
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel*
_output_shapes

:*
dtype0
r
dense_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_55/bias
k
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
_output_shapes
:*
dtype0
z
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_56/kernel
s
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel*
_output_shapes

:*
dtype0
r
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_56/bias
k
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv1d_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_36/kernel/m

+Adam/conv1d_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/kernel/m*"
_output_shapes
:*
dtype0

Adam/conv1d_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_36/bias/m
{
)Adam/conv1d_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_36/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_36/gamma/m

7Adam/batch_normalization_36/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_36/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_36/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_36/beta/m

6Adam/batch_normalization_36/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_36/beta/m*
_output_shapes
:*
dtype0

Adam/conv1d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_37/kernel/m

+Adam/conv1d_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/kernel/m*"
_output_shapes
:*
dtype0

Adam/conv1d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_37/bias/m
{
)Adam/conv1d_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_37/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_37/gamma/m

7Adam/batch_normalization_37/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_37/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_37/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_37/beta/m

6Adam/batch_normalization_37/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_37/beta/m*
_output_shapes
:*
dtype0

Adam/dense_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ø*'
shared_nameAdam/dense_54/kernel/m

*Adam/dense_54/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/m*
_output_shapes
:	Ø*
dtype0

Adam/dense_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_54/bias/m
y
(Adam/dense_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/m*
_output_shapes
:*
dtype0

Adam/dense_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_55/kernel/m

*Adam/dense_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_55/bias/m
y
(Adam/dense_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/m*
_output_shapes
:*
dtype0

Adam/dense_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_56/kernel/m

*Adam/dense_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_56/bias/m
y
(Adam/dense_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/m*
_output_shapes
:*
dtype0

Adam/conv1d_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_36/kernel/v

+Adam/conv1d_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/kernel/v*"
_output_shapes
:*
dtype0

Adam/conv1d_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_36/bias/v
{
)Adam/conv1d_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_36/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_36/gamma/v

7Adam/batch_normalization_36/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_36/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_36/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_36/beta/v

6Adam/batch_normalization_36/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_36/beta/v*
_output_shapes
:*
dtype0

Adam/conv1d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_37/kernel/v

+Adam/conv1d_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/kernel/v*"
_output_shapes
:*
dtype0

Adam/conv1d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_37/bias/v
{
)Adam/conv1d_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_37/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_37/gamma/v

7Adam/batch_normalization_37/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_37/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_37/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_37/beta/v

6Adam/batch_normalization_37/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_37/beta/v*
_output_shapes
:*
dtype0

Adam/dense_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ø*'
shared_nameAdam/dense_54/kernel/v

*Adam/dense_54/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/v*
_output_shapes
:	Ø*
dtype0

Adam/dense_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_54/bias/v
y
(Adam/dense_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/v*
_output_shapes
:*
dtype0

Adam/dense_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_55/kernel/v

*Adam/dense_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_55/bias/v
y
(Adam/dense_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/v*
_output_shapes
:*
dtype0

Adam/dense_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_56/kernel/v

*Adam/dense_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_56/bias/v
y
(Adam/dense_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
£[
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÞZ
valueÔZBÑZ BÊZ
Å
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api

axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
 	variables
!	keras_api
R
"trainable_variables
#regularization_losses
$	variables
%	keras_api
h

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api

,axis
	-gamma
.beta
/moving_mean
0moving_variance
1trainable_variables
2regularization_losses
3	variables
4	keras_api
R
5trainable_variables
6regularization_losses
7	variables
8	keras_api
R
9trainable_variables
:regularization_losses
;	variables
<	keras_api
h

=kernel
>bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
R
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
h

Gkernel
Hbias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
R
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
h

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
Ø

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate
[iterm¨m©mªm«&m¬'m­-m®.m¯=m°>m±Gm²Hm³Qm´Rmµv¶v·v¸v¹&vº'v»-v¼.v½=v¾>v¿GvÀHvÁQvÂRvÃ
 

0
1
2
3
4
5
&6
'7
-8
.9
/10
011
=12
>13
G14
H15
Q16
R17
f
0
1
2
3
&4
'5
-6
.7
=8
>9
G10
H11
Q12
R13
­
\layer_regularization_losses
regularization_losses

]layers
^non_trainable_variables
	variables
_layer_metrics
trainable_variables
`metrics
 
\Z
VARIABLE_VALUEconv1d_36/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_36/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
alayer_regularization_losses
regularization_losses

blayers
	variables
clayer_metrics
dnon_trainable_variables
emetrics
 
ge
VARIABLE_VALUEbatch_normalization_36/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_36/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_36/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_36/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
2
3
­
trainable_variables
flayer_regularization_losses
regularization_losses

glayers
 	variables
hlayer_metrics
inon_trainable_variables
jmetrics
 
 
 
­
"trainable_variables
klayer_regularization_losses
#regularization_losses

llayers
$	variables
mlayer_metrics
nnon_trainable_variables
ometrics
\Z
VARIABLE_VALUEconv1d_37/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_37/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
­
(trainable_variables
player_regularization_losses
)regularization_losses

qlayers
*	variables
rlayer_metrics
snon_trainable_variables
tmetrics
 
ge
VARIABLE_VALUEbatch_normalization_37/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_37/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_37/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_37/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
/2
03
­
1trainable_variables
ulayer_regularization_losses
2regularization_losses

vlayers
3	variables
wlayer_metrics
xnon_trainable_variables
ymetrics
 
 
 
­
5trainable_variables
zlayer_regularization_losses
6regularization_losses

{layers
7	variables
|layer_metrics
}non_trainable_variables
~metrics
 
 
 
±
9trainable_variables
layer_regularization_losses
:regularization_losses
layers
;	variables
layer_metrics
non_trainable_variables
metrics
[Y
VARIABLE_VALUEdense_54/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_54/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
 

=0
>1
²
?trainable_variables
 layer_regularization_losses
@regularization_losses
layers
A	variables
layer_metrics
non_trainable_variables
metrics
 
 
 
²
Ctrainable_variables
 layer_regularization_losses
Dregularization_losses
layers
E	variables
layer_metrics
non_trainable_variables
metrics
[Y
VARIABLE_VALUEdense_55/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_55/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
 

G0
H1
²
Itrainable_variables
 layer_regularization_losses
Jregularization_losses
layers
K	variables
layer_metrics
non_trainable_variables
metrics
 
 
 
²
Mtrainable_variables
 layer_regularization_losses
Nregularization_losses
layers
O	variables
layer_metrics
non_trainable_variables
metrics
[Y
VARIABLE_VALUEdense_56/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_56/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
 

Q0
R1
²
Strainable_variables
 layer_regularization_losses
Tregularization_losses
layers
U	variables
layer_metrics
non_trainable_variables
metrics
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
 
V
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

0
1
/2
03
 

0
1
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

/0
01
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

 count
¡	variables
¢	keras_api
I

£total

¤count
¥
_fn_kwargs
¦	variables
§	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
 1

¡	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

£0
¤1

¦	variables
}
VARIABLE_VALUEAdam/conv1d_36/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_36/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_36/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_36/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_37/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_37/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_37/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_37/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_54/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_54/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_55/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_55/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_56/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_56/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_36/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_36/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_36/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_36/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_37/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_37/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_37/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_37/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_54/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_54/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_55/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_55/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_56/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_56/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv1d_36_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿH

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_36_inputconv1d_36/kernelconv1d_36/bias&batch_normalization_36/moving_variancebatch_normalization_36/gamma"batch_normalization_36/moving_meanbatch_normalization_36/betaconv1d_37/kernelconv1d_37/bias&batch_normalization_37/moving_variancebatch_normalization_37/gamma"batch_normalization_37/moving_meanbatch_normalization_37/betadense_54/kerneldense_54/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_398644
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_36/kernel/Read/ReadVariableOp"conv1d_36/bias/Read/ReadVariableOp0batch_normalization_36/gamma/Read/ReadVariableOp/batch_normalization_36/beta/Read/ReadVariableOp6batch_normalization_36/moving_mean/Read/ReadVariableOp:batch_normalization_36/moving_variance/Read/ReadVariableOp$conv1d_37/kernel/Read/ReadVariableOp"conv1d_37/bias/Read/ReadVariableOp0batch_normalization_37/gamma/Read/ReadVariableOp/batch_normalization_37/beta/Read/ReadVariableOp6batch_normalization_37/moving_mean/Read/ReadVariableOp:batch_normalization_37/moving_variance/Read/ReadVariableOp#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOp#dense_55/kernel/Read/ReadVariableOp!dense_55/bias/Read/ReadVariableOp#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv1d_36/kernel/m/Read/ReadVariableOp)Adam/conv1d_36/bias/m/Read/ReadVariableOp7Adam/batch_normalization_36/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_36/beta/m/Read/ReadVariableOp+Adam/conv1d_37/kernel/m/Read/ReadVariableOp)Adam/conv1d_37/bias/m/Read/ReadVariableOp7Adam/batch_normalization_37/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_37/beta/m/Read/ReadVariableOp*Adam/dense_54/kernel/m/Read/ReadVariableOp(Adam/dense_54/bias/m/Read/ReadVariableOp*Adam/dense_55/kernel/m/Read/ReadVariableOp(Adam/dense_55/bias/m/Read/ReadVariableOp*Adam/dense_56/kernel/m/Read/ReadVariableOp(Adam/dense_56/bias/m/Read/ReadVariableOp+Adam/conv1d_36/kernel/v/Read/ReadVariableOp)Adam/conv1d_36/bias/v/Read/ReadVariableOp7Adam/batch_normalization_36/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_36/beta/v/Read/ReadVariableOp+Adam/conv1d_37/kernel/v/Read/ReadVariableOp)Adam/conv1d_37/bias/v/Read/ReadVariableOp7Adam/batch_normalization_37/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_37/beta/v/Read/ReadVariableOp*Adam/dense_54/kernel/v/Read/ReadVariableOp(Adam/dense_54/bias/v/Read/ReadVariableOp*Adam/dense_55/kernel/v/Read/ReadVariableOp(Adam/dense_55/bias/v/Read/ReadVariableOp*Adam/dense_56/kernel/v/Read/ReadVariableOp(Adam/dense_56/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *(
f#R!
__inference__traced_save_399649
Í
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_36/kernelconv1d_36/biasbatch_normalization_36/gammabatch_normalization_36/beta"batch_normalization_36/moving_mean&batch_normalization_36/moving_varianceconv1d_37/kernelconv1d_37/biasbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_variancedense_54/kerneldense_54/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1Adam/conv1d_36/kernel/mAdam/conv1d_36/bias/m#Adam/batch_normalization_36/gamma/m"Adam/batch_normalization_36/beta/mAdam/conv1d_37/kernel/mAdam/conv1d_37/bias/m#Adam/batch_normalization_37/gamma/m"Adam/batch_normalization_37/beta/mAdam/dense_54/kernel/mAdam/dense_54/bias/mAdam/dense_55/kernel/mAdam/dense_55/bias/mAdam/dense_56/kernel/mAdam/dense_56/bias/mAdam/conv1d_36/kernel/vAdam/conv1d_36/bias/v#Adam/batch_normalization_36/gamma/v"Adam/batch_normalization_36/beta/vAdam/conv1d_37/kernel/vAdam/conv1d_37/bias/v#Adam/batch_normalization_37/gamma/v"Adam/batch_normalization_37/beta/vAdam/dense_54/kernel/vAdam/dense_54/bias/vAdam/dense_55/kernel/vAdam/dense_55/bias/vAdam/dense_56/kernel/vAdam/dense_56/bias/v*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_399824²
è
h
L__inference_max_pooling1d_36_layer_call_and_return_conditional_losses_397785

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDims°
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó

*__inference_conv1d_37_layer_call_fn_399172

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_37_layer_call_and_return_conditional_losses_3980902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ$::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
§
d
+__inference_dropout_36_layer_call_fn_399389

inputs
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_3982512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_398161

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ$::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
ç	
Ý
D__inference_dense_54_layer_call_and_return_conditional_losses_398223

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
:
é
I__inference_sequential_18_layer_call_and_return_conditional_losses_398554

inputs
conv1d_36_398505
conv1d_36_398507!
batch_normalization_36_398510!
batch_normalization_36_398512!
batch_normalization_36_398514!
batch_normalization_36_398516
conv1d_37_398520
conv1d_37_398522!
batch_normalization_37_398525!
batch_normalization_37_398527!
batch_normalization_37_398529!
batch_normalization_37_398531
dense_54_398536
dense_54_398538
dense_55_398542
dense_55_398544
dense_56_398548
dense_56_398550
identity¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢!conv1d_36/StatefulPartitionedCall¢!conv1d_37/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall¢ dense_56/StatefulPartitionedCall¢
!conv1d_36/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_36_398505conv1d_36_398507*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_36_layer_call_and_return_conditional_losses_3979662#
!conv1d_36/StatefulPartitionedCallÉ
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv1d_36/StatefulPartitionedCall:output:0batch_normalization_36_398510batch_normalization_36_398512batch_normalization_36_398514batch_normalization_36_398516*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_39803720
.batch_normalization_36/StatefulPartitionedCall¦
 max_pooling1d_36/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_36_layer_call_and_return_conditional_losses_3977852"
 max_pooling1d_36/PartitionedCallÅ
!conv1d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_36/PartitionedCall:output:0conv1d_37_398520conv1d_37_398522*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_37_layer_call_and_return_conditional_losses_3980902#
!conv1d_37/StatefulPartitionedCallÉ
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv1d_37/StatefulPartitionedCall:output:0batch_normalization_37_398525batch_normalization_37_398527batch_normalization_37_398529batch_normalization_37_398531*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_39816120
.batch_normalization_37/StatefulPartitionedCall¦
 max_pooling1d_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_37_layer_call_and_return_conditional_losses_3979402"
 max_pooling1d_37/PartitionedCall
flatten_18/PartitionedCallPartitionedCall)max_pooling1d_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_3982042
flatten_18/PartitionedCall¶
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_54_398536dense_54_398538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_3982232"
 dense_54/StatefulPartitionedCall
dropout_36/PartitionedCallPartitionedCall)dense_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_3982562
dropout_36/PartitionedCall¶
 dense_55/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0dense_55_398542dense_55_398544*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_55_layer_call_and_return_conditional_losses_3982802"
 dense_55/StatefulPartitionedCall
dropout_37/PartitionedCallPartitionedCall)dense_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_3983132
dropout_37/PartitionedCall¶
 dense_56/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_56_398548dense_56_398550*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_3983372"
 dense_56/StatefulPartitionedCall
IdentityIdentity)dense_56/StatefulPartitionedCall:output:0/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall"^conv1d_36/StatefulPartitionedCall"^conv1d_37/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2F
!conv1d_36/StatefulPartitionedCall!conv1d_36/StatefulPartitionedCall2F
!conv1d_37/StatefulPartitionedCall!conv1d_37/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
ó

*__inference_conv1d_36_layer_call_fn_398983

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_36_layer_call_and_return_conditional_losses_3979662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿH::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs


R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399039

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
F__inference_dropout_36_layer_call_and_return_conditional_losses_399379

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_398313

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ú
.__inference_sequential_18_layer_call_fn_398958

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_sequential_18_layer_call_and_return_conditional_losses_3985542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
É
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_398256

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ø
E__inference_conv1d_37_layer_call_and_return_conditional_losses_399163

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
§
d
+__inference_dropout_37_layer_call_fn_399436

inputs
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_3983082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
ª
7__inference_batch_normalization_37_layer_call_fn_399254

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_3981612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ$::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
ë

ù
$__inference_signature_wrapper_398644
conv1d_36_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallconv1d_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_3976362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
)
_user_specified_nameconv1d_36_input
ãé

"__inference__traced_restore_399824
file_prefix%
!assignvariableop_conv1d_36_kernel%
!assignvariableop_1_conv1d_36_bias3
/assignvariableop_2_batch_normalization_36_gamma2
.assignvariableop_3_batch_normalization_36_beta9
5assignvariableop_4_batch_normalization_36_moving_mean=
9assignvariableop_5_batch_normalization_36_moving_variance'
#assignvariableop_6_conv1d_37_kernel%
!assignvariableop_7_conv1d_37_bias3
/assignvariableop_8_batch_normalization_37_gamma2
.assignvariableop_9_batch_normalization_37_beta:
6assignvariableop_10_batch_normalization_37_moving_mean>
:assignvariableop_11_batch_normalization_37_moving_variance'
#assignvariableop_12_dense_54_kernel%
!assignvariableop_13_dense_54_bias'
#assignvariableop_14_dense_55_kernel%
!assignvariableop_15_dense_55_bias'
#assignvariableop_16_dense_56_kernel%
!assignvariableop_17_dense_56_bias
assignvariableop_18_beta_1
assignvariableop_19_beta_2
assignvariableop_20_decay%
!assignvariableop_21_learning_rate!
assignvariableop_22_adam_iter
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1/
+assignvariableop_27_adam_conv1d_36_kernel_m-
)assignvariableop_28_adam_conv1d_36_bias_m;
7assignvariableop_29_adam_batch_normalization_36_gamma_m:
6assignvariableop_30_adam_batch_normalization_36_beta_m/
+assignvariableop_31_adam_conv1d_37_kernel_m-
)assignvariableop_32_adam_conv1d_37_bias_m;
7assignvariableop_33_adam_batch_normalization_37_gamma_m:
6assignvariableop_34_adam_batch_normalization_37_beta_m.
*assignvariableop_35_adam_dense_54_kernel_m,
(assignvariableop_36_adam_dense_54_bias_m.
*assignvariableop_37_adam_dense_55_kernel_m,
(assignvariableop_38_adam_dense_55_bias_m.
*assignvariableop_39_adam_dense_56_kernel_m,
(assignvariableop_40_adam_dense_56_bias_m/
+assignvariableop_41_adam_conv1d_36_kernel_v-
)assignvariableop_42_adam_conv1d_36_bias_v;
7assignvariableop_43_adam_batch_normalization_36_gamma_v:
6assignvariableop_44_adam_batch_normalization_36_beta_v/
+assignvariableop_45_adam_conv1d_37_kernel_v-
)assignvariableop_46_adam_conv1d_37_bias_v;
7assignvariableop_47_adam_batch_normalization_37_gamma_v:
6assignvariableop_48_adam_batch_normalization_37_beta_v.
*assignvariableop_49_adam_dense_54_kernel_v,
(assignvariableop_50_adam_dense_54_bias_v.
*assignvariableop_51_adam_dense_55_kernel_v,
(assignvariableop_52_adam_dense_55_bias_v.
*assignvariableop_53_adam_dense_56_kernel_v,
(assignvariableop_54_adam_dense_56_bias_v
identity_56¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ð
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*ü
valueòBï8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÿ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÆ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ö
_output_shapesã
à::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_36_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_36_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2´
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_36_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3³
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_36_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4º
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_36_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¾
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_36_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv1d_37_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv1d_37_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8´
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_37_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_37_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¾
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_37_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Â
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_37_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_54_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_54_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_55_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_55_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_56_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_56_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¢
AssignVariableOp_18AssignVariableOpassignvariableop_18_beta_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¢
AssignVariableOp_19AssignVariableOpassignvariableop_19_beta_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¡
AssignVariableOp_20AssignVariableOpassignvariableop_20_decayIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22¥
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¡
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¡
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv1d_36_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv1d_36_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¿
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_batch_normalization_36_gamma_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¾
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_batch_normalization_36_beta_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv1d_37_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv1d_37_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¿
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_batch_normalization_37_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¾
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_batch_normalization_37_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_54_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_54_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_55_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_55_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39²
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_56_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40°
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_56_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41³
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv1d_36_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42±
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv1d_36_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¿
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_36_gamma_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¾
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_36_beta_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45³
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv1d_37_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46±
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv1d_37_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¿
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_batch_normalization_37_gamma_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¾
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_batch_normalization_37_beta_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49²
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_54_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50°
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_54_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51²
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_55_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52°
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_55_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53²
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_56_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54°
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_56_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55

Identity_56IdentityIdentity_55:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_56"#
identity_56Identity_56:output:0*ó
_input_shapesá
Þ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
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
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ð
ª
7__inference_batch_normalization_37_layer_call_fn_399323

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_3978872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä	
Ý
D__inference_dense_55_layer_call_and_return_conditional_losses_399405

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


.__inference_sequential_18_layer_call_fn_398500
conv1d_36_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallconv1d_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_sequential_18_layer_call_and_return_conditional_losses_3984612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
)
_user_specified_nameconv1d_36_input

ø
E__inference_conv1d_36_layer_call_and_return_conditional_losses_398974

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿH::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
ä	
Ý
D__inference_dense_55_layer_call_and_return_conditional_losses_398280

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µp

__inference__traced_save_399649
file_prefix/
+savev2_conv1d_36_kernel_read_readvariableop-
)savev2_conv1d_36_bias_read_readvariableop;
7savev2_batch_normalization_36_gamma_read_readvariableop:
6savev2_batch_normalization_36_beta_read_readvariableopA
=savev2_batch_normalization_36_moving_mean_read_readvariableopE
Asavev2_batch_normalization_36_moving_variance_read_readvariableop/
+savev2_conv1d_37_kernel_read_readvariableop-
)savev2_conv1d_37_bias_read_readvariableop;
7savev2_batch_normalization_37_gamma_read_readvariableop:
6savev2_batch_normalization_37_beta_read_readvariableopA
=savev2_batch_normalization_37_moving_mean_read_readvariableopE
Asavev2_batch_normalization_37_moving_variance_read_readvariableop.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop.
*savev2_dense_55_kernel_read_readvariableop,
(savev2_dense_55_bias_read_readvariableop.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv1d_36_kernel_m_read_readvariableop4
0savev2_adam_conv1d_36_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_36_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_36_beta_m_read_readvariableop6
2savev2_adam_conv1d_37_kernel_m_read_readvariableop4
0savev2_adam_conv1d_37_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_37_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_37_beta_m_read_readvariableop5
1savev2_adam_dense_54_kernel_m_read_readvariableop3
/savev2_adam_dense_54_bias_m_read_readvariableop5
1savev2_adam_dense_55_kernel_m_read_readvariableop3
/savev2_adam_dense_55_bias_m_read_readvariableop5
1savev2_adam_dense_56_kernel_m_read_readvariableop3
/savev2_adam_dense_56_bias_m_read_readvariableop6
2savev2_adam_conv1d_36_kernel_v_read_readvariableop4
0savev2_adam_conv1d_36_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_36_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_36_beta_v_read_readvariableop6
2savev2_adam_conv1d_37_kernel_v_read_readvariableop4
0savev2_adam_conv1d_37_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_37_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_37_beta_v_read_readvariableop5
1savev2_adam_dense_54_kernel_v_read_readvariableop3
/savev2_adam_dense_54_bias_v_read_readvariableop5
1savev2_adam_dense_55_kernel_v_read_readvariableop3
/savev2_adam_dense_55_bias_v_read_readvariableop5
1savev2_adam_dense_56_kernel_v_read_readvariableop3
/savev2_adam_dense_56_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameê
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*ü
valueòBï8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesù
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¹
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_36_kernel_read_readvariableop)savev2_conv1d_36_bias_read_readvariableop7savev2_batch_normalization_36_gamma_read_readvariableop6savev2_batch_normalization_36_beta_read_readvariableop=savev2_batch_normalization_36_moving_mean_read_readvariableopAsavev2_batch_normalization_36_moving_variance_read_readvariableop+savev2_conv1d_37_kernel_read_readvariableop)savev2_conv1d_37_bias_read_readvariableop7savev2_batch_normalization_37_gamma_read_readvariableop6savev2_batch_normalization_37_beta_read_readvariableop=savev2_batch_normalization_37_moving_mean_read_readvariableopAsavev2_batch_normalization_37_moving_variance_read_readvariableop*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop*savev2_dense_55_kernel_read_readvariableop(savev2_dense_55_bias_read_readvariableop*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv1d_36_kernel_m_read_readvariableop0savev2_adam_conv1d_36_bias_m_read_readvariableop>savev2_adam_batch_normalization_36_gamma_m_read_readvariableop=savev2_adam_batch_normalization_36_beta_m_read_readvariableop2savev2_adam_conv1d_37_kernel_m_read_readvariableop0savev2_adam_conv1d_37_bias_m_read_readvariableop>savev2_adam_batch_normalization_37_gamma_m_read_readvariableop=savev2_adam_batch_normalization_37_beta_m_read_readvariableop1savev2_adam_dense_54_kernel_m_read_readvariableop/savev2_adam_dense_54_bias_m_read_readvariableop1savev2_adam_dense_55_kernel_m_read_readvariableop/savev2_adam_dense_55_bias_m_read_readvariableop1savev2_adam_dense_56_kernel_m_read_readvariableop/savev2_adam_dense_56_bias_m_read_readvariableop2savev2_adam_conv1d_36_kernel_v_read_readvariableop0savev2_adam_conv1d_36_bias_v_read_readvariableop>savev2_adam_batch_normalization_36_gamma_v_read_readvariableop=savev2_adam_batch_normalization_36_beta_v_read_readvariableop2savev2_adam_conv1d_37_kernel_v_read_readvariableop0savev2_adam_conv1d_37_bias_v_read_readvariableop>savev2_adam_batch_normalization_37_gamma_v_read_readvariableop=savev2_adam_batch_normalization_37_beta_v_read_readvariableop1savev2_adam_dense_54_kernel_v_read_readvariableop/savev2_adam_dense_54_bias_v_read_readvariableop1savev2_adam_dense_55_kernel_v_read_readvariableop/savev2_adam_dense_55_bias_v_read_readvariableop1savev2_adam_dense_56_kernel_v_read_readvariableop/savev2_adam_dense_56_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :::::::::::::	Ø:::::: : : : : : : : : :::::::::	Ø::::::::::::::	Ø:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	Ø: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::( $
"
_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
::%$!

_output_shapes
:	Ø: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::(*$
"
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::(.$
"
_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
::%2!

_output_shapes
:	Ø: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::8

_output_shapes
: 
½0
É
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399208

inputs
assignmovingavg_399183
assignmovingavg_1_399189)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/399183*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_399183*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/399183*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/399183*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_399183AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/399183*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/399189*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_399189*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399189*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399189*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_399189AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/399189*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ$::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
É
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_399431

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô0
É
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399019

inputs
assignmovingavg_398994
assignmovingavg_1_399000)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/398994*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_398994*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/398994*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/398994*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_398994AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/398994*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/399000*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_399000*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399000*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399000*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_399000AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/399000*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
F__inference_dropout_36_layer_call_and_return_conditional_losses_398251

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399228

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ$::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs


R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399310

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½0
É
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_398017

inputs
assignmovingavg_397992
assignmovingavg_1_397998)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/397992*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_397992*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/397992*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/397992*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_397992AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/397992*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/397998*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_397998*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/397998*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/397998*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_397998AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/397998*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿH::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
á
~
)__inference_dense_56_layer_call_fn_399461

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_3983372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_399384

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
³
!__inference__wrapped_model_397636
conv1d_36_inputG
Csequential_18_conv1d_36_conv1d_expanddims_1_readvariableop_resource;
7sequential_18_conv1d_36_biasadd_readvariableop_resourceJ
Fsequential_18_batch_normalization_36_batchnorm_readvariableop_resourceN
Jsequential_18_batch_normalization_36_batchnorm_mul_readvariableop_resourceL
Hsequential_18_batch_normalization_36_batchnorm_readvariableop_1_resourceL
Hsequential_18_batch_normalization_36_batchnorm_readvariableop_2_resourceG
Csequential_18_conv1d_37_conv1d_expanddims_1_readvariableop_resource;
7sequential_18_conv1d_37_biasadd_readvariableop_resourceJ
Fsequential_18_batch_normalization_37_batchnorm_readvariableop_resourceN
Jsequential_18_batch_normalization_37_batchnorm_mul_readvariableop_resourceL
Hsequential_18_batch_normalization_37_batchnorm_readvariableop_1_resourceL
Hsequential_18_batch_normalization_37_batchnorm_readvariableop_2_resource9
5sequential_18_dense_54_matmul_readvariableop_resource:
6sequential_18_dense_54_biasadd_readvariableop_resource9
5sequential_18_dense_55_matmul_readvariableop_resource:
6sequential_18_dense_55_biasadd_readvariableop_resource9
5sequential_18_dense_56_matmul_readvariableop_resource:
6sequential_18_dense_56_biasadd_readvariableop_resource
identity¢=sequential_18/batch_normalization_36/batchnorm/ReadVariableOp¢?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_1¢?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_2¢Asequential_18/batch_normalization_36/batchnorm/mul/ReadVariableOp¢=sequential_18/batch_normalization_37/batchnorm/ReadVariableOp¢?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_1¢?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_2¢Asequential_18/batch_normalization_37/batchnorm/mul/ReadVariableOp¢.sequential_18/conv1d_36/BiasAdd/ReadVariableOp¢:sequential_18/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp¢.sequential_18/conv1d_37/BiasAdd/ReadVariableOp¢:sequential_18/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp¢-sequential_18/dense_54/BiasAdd/ReadVariableOp¢,sequential_18/dense_54/MatMul/ReadVariableOp¢-sequential_18/dense_55/BiasAdd/ReadVariableOp¢,sequential_18/dense_55/MatMul/ReadVariableOp¢-sequential_18/dense_56/BiasAdd/ReadVariableOp¢,sequential_18/dense_56/MatMul/ReadVariableOp©
-sequential_18/conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2/
-sequential_18/conv1d_36/conv1d/ExpandDims/dimç
)sequential_18/conv1d_36/conv1d/ExpandDims
ExpandDimsconv1d_36_input6sequential_18/conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2+
)sequential_18/conv1d_36/conv1d/ExpandDims
:sequential_18/conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_18_conv1d_36_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02<
:sequential_18/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp¤
/sequential_18/conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_18/conv1d_36/conv1d/ExpandDims_1/dim
+sequential_18/conv1d_36/conv1d/ExpandDims_1
ExpandDimsBsequential_18/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_18/conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2-
+sequential_18/conv1d_36/conv1d/ExpandDims_1
sequential_18/conv1d_36/conv1dConv2D2sequential_18/conv1d_36/conv1d/ExpandDims:output:04sequential_18/conv1d_36/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
2 
sequential_18/conv1d_36/conv1dÚ
&sequential_18/conv1d_36/conv1d/SqueezeSqueeze'sequential_18/conv1d_36/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2(
&sequential_18/conv1d_36/conv1d/SqueezeÔ
.sequential_18/conv1d_36/BiasAdd/ReadVariableOpReadVariableOp7sequential_18_conv1d_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_18/conv1d_36/BiasAdd/ReadVariableOpì
sequential_18/conv1d_36/BiasAddBiasAdd/sequential_18/conv1d_36/conv1d/Squeeze:output:06sequential_18/conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2!
sequential_18/conv1d_36/BiasAdd¤
sequential_18/conv1d_36/ReluRelu(sequential_18/conv1d_36/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
sequential_18/conv1d_36/Relu
=sequential_18/batch_normalization_36/batchnorm/ReadVariableOpReadVariableOpFsequential_18_batch_normalization_36_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_18/batch_normalization_36/batchnorm/ReadVariableOp±
4sequential_18/batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4sequential_18/batch_normalization_36/batchnorm/add/y
2sequential_18/batch_normalization_36/batchnorm/addAddV2Esequential_18/batch_normalization_36/batchnorm/ReadVariableOp:value:0=sequential_18/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_18/batch_normalization_36/batchnorm/addÒ
4sequential_18/batch_normalization_36/batchnorm/RsqrtRsqrt6sequential_18/batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_18/batch_normalization_36/batchnorm/Rsqrt
Asequential_18/batch_normalization_36/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_18_batch_normalization_36_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_18/batch_normalization_36/batchnorm/mul/ReadVariableOp
2sequential_18/batch_normalization_36/batchnorm/mulMul8sequential_18/batch_normalization_36/batchnorm/Rsqrt:y:0Isequential_18/batch_normalization_36/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_18/batch_normalization_36/batchnorm/mul
4sequential_18/batch_normalization_36/batchnorm/mul_1Mul*sequential_18/conv1d_36/Relu:activations:06sequential_18/batch_normalization_36/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH26
4sequential_18/batch_normalization_36/batchnorm/mul_1
?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_18_batch_normalization_36_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_1
4sequential_18/batch_normalization_36/batchnorm/mul_2MulGsequential_18/batch_normalization_36/batchnorm/ReadVariableOp_1:value:06sequential_18/batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_18/batch_normalization_36/batchnorm/mul_2
?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_18_batch_normalization_36_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_2
2sequential_18/batch_normalization_36/batchnorm/subSubGsequential_18/batch_normalization_36/batchnorm/ReadVariableOp_2:value:08sequential_18/batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_18/batch_normalization_36/batchnorm/sub
4sequential_18/batch_normalization_36/batchnorm/add_1AddV28sequential_18/batch_normalization_36/batchnorm/mul_1:z:06sequential_18/batch_normalization_36/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH26
4sequential_18/batch_normalization_36/batchnorm/add_1 
-sequential_18/max_pooling1d_36/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_18/max_pooling1d_36/ExpandDims/dim
)sequential_18/max_pooling1d_36/ExpandDims
ExpandDims8sequential_18/batch_normalization_36/batchnorm/add_1:z:06sequential_18/max_pooling1d_36/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2+
)sequential_18/max_pooling1d_36/ExpandDimsû
&sequential_18/max_pooling1d_36/MaxPoolMaxPool2sequential_18/max_pooling1d_36/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
ksize
*
paddingSAME*
strides
2(
&sequential_18/max_pooling1d_36/MaxPoolÙ
&sequential_18/max_pooling1d_36/SqueezeSqueeze/sequential_18/max_pooling1d_36/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
squeeze_dims
2(
&sequential_18/max_pooling1d_36/Squeeze©
-sequential_18/conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2/
-sequential_18/conv1d_37/conv1d/ExpandDims/dim
)sequential_18/conv1d_37/conv1d/ExpandDims
ExpandDims/sequential_18/max_pooling1d_36/Squeeze:output:06sequential_18/conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2+
)sequential_18/conv1d_37/conv1d/ExpandDims
:sequential_18/conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_18_conv1d_37_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02<
:sequential_18/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp¤
/sequential_18/conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_18/conv1d_37/conv1d/ExpandDims_1/dim
+sequential_18/conv1d_37/conv1d/ExpandDims_1
ExpandDimsBsequential_18/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_18/conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2-
+sequential_18/conv1d_37/conv1d/ExpandDims_1
sequential_18/conv1d_37/conv1dConv2D2sequential_18/conv1d_37/conv1d/ExpandDims:output:04sequential_18/conv1d_37/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
paddingSAME*
strides
2 
sequential_18/conv1d_37/conv1dÚ
&sequential_18/conv1d_37/conv1d/SqueezeSqueeze'sequential_18/conv1d_37/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2(
&sequential_18/conv1d_37/conv1d/SqueezeÔ
.sequential_18/conv1d_37/BiasAdd/ReadVariableOpReadVariableOp7sequential_18_conv1d_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_18/conv1d_37/BiasAdd/ReadVariableOpì
sequential_18/conv1d_37/BiasAddBiasAdd/sequential_18/conv1d_37/conv1d/Squeeze:output:06sequential_18/conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2!
sequential_18/conv1d_37/BiasAdd¤
sequential_18/conv1d_37/ReluRelu(sequential_18/conv1d_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
sequential_18/conv1d_37/Relu
=sequential_18/batch_normalization_37/batchnorm/ReadVariableOpReadVariableOpFsequential_18_batch_normalization_37_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_18/batch_normalization_37/batchnorm/ReadVariableOp±
4sequential_18/batch_normalization_37/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4sequential_18/batch_normalization_37/batchnorm/add/y
2sequential_18/batch_normalization_37/batchnorm/addAddV2Esequential_18/batch_normalization_37/batchnorm/ReadVariableOp:value:0=sequential_18/batch_normalization_37/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_18/batch_normalization_37/batchnorm/addÒ
4sequential_18/batch_normalization_37/batchnorm/RsqrtRsqrt6sequential_18/batch_normalization_37/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_18/batch_normalization_37/batchnorm/Rsqrt
Asequential_18/batch_normalization_37/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_18_batch_normalization_37_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_18/batch_normalization_37/batchnorm/mul/ReadVariableOp
2sequential_18/batch_normalization_37/batchnorm/mulMul8sequential_18/batch_normalization_37/batchnorm/Rsqrt:y:0Isequential_18/batch_normalization_37/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_18/batch_normalization_37/batchnorm/mul
4sequential_18/batch_normalization_37/batchnorm/mul_1Mul*sequential_18/conv1d_37/Relu:activations:06sequential_18/batch_normalization_37/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$26
4sequential_18/batch_normalization_37/batchnorm/mul_1
?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_18_batch_normalization_37_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_1
4sequential_18/batch_normalization_37/batchnorm/mul_2MulGsequential_18/batch_normalization_37/batchnorm/ReadVariableOp_1:value:06sequential_18/batch_normalization_37/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_18/batch_normalization_37/batchnorm/mul_2
?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_18_batch_normalization_37_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_2
2sequential_18/batch_normalization_37/batchnorm/subSubGsequential_18/batch_normalization_37/batchnorm/ReadVariableOp_2:value:08sequential_18/batch_normalization_37/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_18/batch_normalization_37/batchnorm/sub
4sequential_18/batch_normalization_37/batchnorm/add_1AddV28sequential_18/batch_normalization_37/batchnorm/mul_1:z:06sequential_18/batch_normalization_37/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$26
4sequential_18/batch_normalization_37/batchnorm/add_1 
-sequential_18/max_pooling1d_37/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_18/max_pooling1d_37/ExpandDims/dim
)sequential_18/max_pooling1d_37/ExpandDims
ExpandDims8sequential_18/batch_normalization_37/batchnorm/add_1:z:06sequential_18/max_pooling1d_37/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2+
)sequential_18/max_pooling1d_37/ExpandDimsû
&sequential_18/max_pooling1d_37/MaxPoolMaxPool2sequential_18/max_pooling1d_37/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2(
&sequential_18/max_pooling1d_37/MaxPoolÙ
&sequential_18/max_pooling1d_37/SqueezeSqueeze/sequential_18/max_pooling1d_37/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2(
&sequential_18/max_pooling1d_37/Squeeze
sequential_18/flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿØ   2 
sequential_18/flatten_18/ConstÜ
 sequential_18/flatten_18/ReshapeReshape/sequential_18/max_pooling1d_37/Squeeze:output:0'sequential_18/flatten_18/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2"
 sequential_18/flatten_18/ReshapeÓ
,sequential_18/dense_54/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_54_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02.
,sequential_18/dense_54/MatMul/ReadVariableOpÛ
sequential_18/dense_54/MatMulMatMul)sequential_18/flatten_18/Reshape:output:04sequential_18/dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_18/dense_54/MatMulÑ
-sequential_18/dense_54/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_18/dense_54/BiasAdd/ReadVariableOpÝ
sequential_18/dense_54/BiasAddBiasAdd'sequential_18/dense_54/MatMul:product:05sequential_18/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_18/dense_54/BiasAdd
sequential_18/dense_54/TanhTanh'sequential_18/dense_54/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_18/dense_54/Tanh¥
!sequential_18/dropout_36/IdentityIdentitysequential_18/dense_54/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_18/dropout_36/IdentityÒ
,sequential_18/dense_55/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_18/dense_55/MatMul/ReadVariableOpÜ
sequential_18/dense_55/MatMulMatMul*sequential_18/dropout_36/Identity:output:04sequential_18/dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_18/dense_55/MatMulÑ
-sequential_18/dense_55/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_18/dense_55/BiasAdd/ReadVariableOpÝ
sequential_18/dense_55/BiasAddBiasAdd'sequential_18/dense_55/MatMul:product:05sequential_18/dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_18/dense_55/BiasAdd
sequential_18/dense_55/TanhTanh'sequential_18/dense_55/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_18/dense_55/Tanh¥
!sequential_18/dropout_37/IdentityIdentitysequential_18/dense_55/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_18/dropout_37/IdentityÒ
,sequential_18/dense_56/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_56_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_18/dense_56/MatMul/ReadVariableOpÜ
sequential_18/dense_56/MatMulMatMul*sequential_18/dropout_37/Identity:output:04sequential_18/dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_18/dense_56/MatMulÑ
-sequential_18/dense_56/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_18/dense_56/BiasAdd/ReadVariableOpÝ
sequential_18/dense_56/BiasAddBiasAdd'sequential_18/dense_56/MatMul:product:05sequential_18/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_18/dense_56/BiasAdd¦
sequential_18/dense_56/SoftmaxSoftmax'sequential_18/dense_56/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_18/dense_56/Softmax	
IdentityIdentity(sequential_18/dense_56/Softmax:softmax:0>^sequential_18/batch_normalization_36/batchnorm/ReadVariableOp@^sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_1@^sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_2B^sequential_18/batch_normalization_36/batchnorm/mul/ReadVariableOp>^sequential_18/batch_normalization_37/batchnorm/ReadVariableOp@^sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_1@^sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_2B^sequential_18/batch_normalization_37/batchnorm/mul/ReadVariableOp/^sequential_18/conv1d_36/BiasAdd/ReadVariableOp;^sequential_18/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp/^sequential_18/conv1d_37/BiasAdd/ReadVariableOp;^sequential_18/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp.^sequential_18/dense_54/BiasAdd/ReadVariableOp-^sequential_18/dense_54/MatMul/ReadVariableOp.^sequential_18/dense_55/BiasAdd/ReadVariableOp-^sequential_18/dense_55/MatMul/ReadVariableOp.^sequential_18/dense_56/BiasAdd/ReadVariableOp-^sequential_18/dense_56/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::2~
=sequential_18/batch_normalization_36/batchnorm/ReadVariableOp=sequential_18/batch_normalization_36/batchnorm/ReadVariableOp2
?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_1?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_12
?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_2?sequential_18/batch_normalization_36/batchnorm/ReadVariableOp_22
Asequential_18/batch_normalization_36/batchnorm/mul/ReadVariableOpAsequential_18/batch_normalization_36/batchnorm/mul/ReadVariableOp2~
=sequential_18/batch_normalization_37/batchnorm/ReadVariableOp=sequential_18/batch_normalization_37/batchnorm/ReadVariableOp2
?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_1?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_12
?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_2?sequential_18/batch_normalization_37/batchnorm/ReadVariableOp_22
Asequential_18/batch_normalization_37/batchnorm/mul/ReadVariableOpAsequential_18/batch_normalization_37/batchnorm/mul/ReadVariableOp2`
.sequential_18/conv1d_36/BiasAdd/ReadVariableOp.sequential_18/conv1d_36/BiasAdd/ReadVariableOp2x
:sequential_18/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:sequential_18/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp2`
.sequential_18/conv1d_37/BiasAdd/ReadVariableOp.sequential_18/conv1d_37/BiasAdd/ReadVariableOp2x
:sequential_18/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:sequential_18/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp2^
-sequential_18/dense_54/BiasAdd/ReadVariableOp-sequential_18/dense_54/BiasAdd/ReadVariableOp2\
,sequential_18/dense_54/MatMul/ReadVariableOp,sequential_18/dense_54/MatMul/ReadVariableOp2^
-sequential_18/dense_55/BiasAdd/ReadVariableOp-sequential_18/dense_55/BiasAdd/ReadVariableOp2\
,sequential_18/dense_55/MatMul/ReadVariableOp,sequential_18/dense_55/MatMul/ReadVariableOp2^
-sequential_18/dense_56/BiasAdd/ReadVariableOp-sequential_18/dense_56/BiasAdd/ReadVariableOp2\
,sequential_18/dense_56/MatMul/ReadVariableOp,sequential_18/dense_56/MatMul/ReadVariableOp:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
)
_user_specified_nameconv1d_36_input

G
+__inference_dropout_36_layer_call_fn_399394

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_3982562
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_398037

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿH::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs

e
F__inference_dropout_37_layer_call_and_return_conditional_losses_399426

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

G
+__inference_dropout_37_layer_call_fn_399441

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_3983132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö	
Ý
D__inference_dense_56_layer_call_and_return_conditional_losses_398337

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô0
É
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_397887

inputs
assignmovingavg_397862
assignmovingavg_1_397868)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/397862*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_397862*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/397862*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/397862*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_397862AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/397862*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/397868*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_397868*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/397868*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/397868*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_397868AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/397868*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô0
É
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399290

inputs
assignmovingavg_399265
assignmovingavg_1_399271)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/399265*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_399265*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/399265*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/399265*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_399265AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/399265*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/399271*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_399271*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399271*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399271*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_399271AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/399271*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
ª
7__inference_batch_normalization_37_layer_call_fn_399336

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_3979202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


.__inference_sequential_18_layer_call_fn_398593
conv1d_36_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallconv1d_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_sequential_18_layer_call_and_return_conditional_losses_3985542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
)
_user_specified_nameconv1d_36_input
þ

ú
.__inference_sequential_18_layer_call_fn_398917

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_sequential_18_layer_call_and_return_conditional_losses_3984612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
Þ
Ú
I__inference_sequential_18_layer_call_and_return_conditional_losses_398876

inputs9
5conv1d_36_conv1d_expanddims_1_readvariableop_resource-
)conv1d_36_biasadd_readvariableop_resource<
8batch_normalization_36_batchnorm_readvariableop_resource@
<batch_normalization_36_batchnorm_mul_readvariableop_resource>
:batch_normalization_36_batchnorm_readvariableop_1_resource>
:batch_normalization_36_batchnorm_readvariableop_2_resource9
5conv1d_37_conv1d_expanddims_1_readvariableop_resource-
)conv1d_37_biasadd_readvariableop_resource<
8batch_normalization_37_batchnorm_readvariableop_resource@
<batch_normalization_37_batchnorm_mul_readvariableop_resource>
:batch_normalization_37_batchnorm_readvariableop_1_resource>
:batch_normalization_37_batchnorm_readvariableop_2_resource+
'dense_54_matmul_readvariableop_resource,
(dense_54_biasadd_readvariableop_resource+
'dense_55_matmul_readvariableop_resource,
(dense_55_biasadd_readvariableop_resource+
'dense_56_matmul_readvariableop_resource,
(dense_56_biasadd_readvariableop_resource
identity¢/batch_normalization_36/batchnorm/ReadVariableOp¢1batch_normalization_36/batchnorm/ReadVariableOp_1¢1batch_normalization_36/batchnorm/ReadVariableOp_2¢3batch_normalization_36/batchnorm/mul/ReadVariableOp¢/batch_normalization_37/batchnorm/ReadVariableOp¢1batch_normalization_37/batchnorm/ReadVariableOp_1¢1batch_normalization_37/batchnorm/ReadVariableOp_2¢3batch_normalization_37/batchnorm/mul/ReadVariableOp¢ conv1d_36/BiasAdd/ReadVariableOp¢,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp¢ conv1d_37/BiasAdd/ReadVariableOp¢,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp¢dense_54/BiasAdd/ReadVariableOp¢dense_54/MatMul/ReadVariableOp¢dense_55/BiasAdd/ReadVariableOp¢dense_55/MatMul/ReadVariableOp¢dense_56/BiasAdd/ReadVariableOp¢dense_56/MatMul/ReadVariableOp
conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_36/conv1d/ExpandDims/dim´
conv1d_36/conv1d/ExpandDims
ExpandDimsinputs(conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
conv1d_36/conv1d/ExpandDimsÖ
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_36_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_36/conv1d/ExpandDims_1/dimß
conv1d_36/conv1d/ExpandDims_1
ExpandDims4conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_36/conv1d/ExpandDims_1Þ
conv1d_36/conv1dConv2D$conv1d_36/conv1d/ExpandDims:output:0&conv1d_36/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
2
conv1d_36/conv1d°
conv1d_36/conv1d/SqueezeSqueezeconv1d_36/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_36/conv1d/Squeezeª
 conv1d_36/BiasAdd/ReadVariableOpReadVariableOp)conv1d_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_36/BiasAdd/ReadVariableOp´
conv1d_36/BiasAddBiasAdd!conv1d_36/conv1d/Squeeze:output:0(conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
conv1d_36/BiasAddz
conv1d_36/ReluReluconv1d_36/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
conv1d_36/Relu×
/batch_normalization_36/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_36_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_36/batchnorm/ReadVariableOp
&batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_36/batchnorm/add/yä
$batch_normalization_36/batchnorm/addAddV27batch_normalization_36/batchnorm/ReadVariableOp:value:0/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_36/batchnorm/add¨
&batch_normalization_36/batchnorm/RsqrtRsqrt(batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_36/batchnorm/Rsqrtã
3batch_normalization_36/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_36_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_36/batchnorm/mul/ReadVariableOpá
$batch_normalization_36/batchnorm/mulMul*batch_normalization_36/batchnorm/Rsqrt:y:0;batch_normalization_36/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_36/batchnorm/mulÕ
&batch_normalization_36/batchnorm/mul_1Mulconv1d_36/Relu:activations:0(batch_normalization_36/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2(
&batch_normalization_36/batchnorm/mul_1Ý
1batch_normalization_36/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_36_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_36/batchnorm/ReadVariableOp_1á
&batch_normalization_36/batchnorm/mul_2Mul9batch_normalization_36/batchnorm/ReadVariableOp_1:value:0(batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_36/batchnorm/mul_2Ý
1batch_normalization_36/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_36_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_36/batchnorm/ReadVariableOp_2ß
$batch_normalization_36/batchnorm/subSub9batch_normalization_36/batchnorm/ReadVariableOp_2:value:0*batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_36/batchnorm/subå
&batch_normalization_36/batchnorm/add_1AddV2*batch_normalization_36/batchnorm/mul_1:z:0(batch_normalization_36/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2(
&batch_normalization_36/batchnorm/add_1
max_pooling1d_36/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_36/ExpandDims/dimØ
max_pooling1d_36/ExpandDims
ExpandDims*batch_normalization_36/batchnorm/add_1:z:0(max_pooling1d_36/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
max_pooling1d_36/ExpandDimsÑ
max_pooling1d_36/MaxPoolMaxPool$max_pooling1d_36/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
ksize
*
paddingSAME*
strides
2
max_pooling1d_36/MaxPool¯
max_pooling1d_36/SqueezeSqueeze!max_pooling1d_36/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
squeeze_dims
2
max_pooling1d_36/Squeeze
conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_37/conv1d/ExpandDims/dimÏ
conv1d_37/conv1d/ExpandDims
ExpandDims!max_pooling1d_36/Squeeze:output:0(conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
conv1d_37/conv1d/ExpandDimsÖ
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_37/conv1d/ExpandDims_1/dimß
conv1d_37/conv1d/ExpandDims_1
ExpandDims4conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_37/conv1d/ExpandDims_1Þ
conv1d_37/conv1dConv2D$conv1d_37/conv1d/ExpandDims:output:0&conv1d_37/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
paddingSAME*
strides
2
conv1d_37/conv1d°
conv1d_37/conv1d/SqueezeSqueezeconv1d_37/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_37/conv1d/Squeezeª
 conv1d_37/BiasAdd/ReadVariableOpReadVariableOp)conv1d_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_37/BiasAdd/ReadVariableOp´
conv1d_37/BiasAddBiasAdd!conv1d_37/conv1d/Squeeze:output:0(conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
conv1d_37/BiasAddz
conv1d_37/ReluReluconv1d_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
conv1d_37/Relu×
/batch_normalization_37/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_37_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_37/batchnorm/ReadVariableOp
&batch_normalization_37/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_37/batchnorm/add/yä
$batch_normalization_37/batchnorm/addAddV27batch_normalization_37/batchnorm/ReadVariableOp:value:0/batch_normalization_37/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/add¨
&batch_normalization_37/batchnorm/RsqrtRsqrt(batch_normalization_37/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/Rsqrtã
3batch_normalization_37/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_37_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_37/batchnorm/mul/ReadVariableOpá
$batch_normalization_37/batchnorm/mulMul*batch_normalization_37/batchnorm/Rsqrt:y:0;batch_normalization_37/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/mulÕ
&batch_normalization_37/batchnorm/mul_1Mulconv1d_37/Relu:activations:0(batch_normalization_37/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2(
&batch_normalization_37/batchnorm/mul_1Ý
1batch_normalization_37/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_37_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_37/batchnorm/ReadVariableOp_1á
&batch_normalization_37/batchnorm/mul_2Mul9batch_normalization_37/batchnorm/ReadVariableOp_1:value:0(batch_normalization_37/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/mul_2Ý
1batch_normalization_37/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_37_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_37/batchnorm/ReadVariableOp_2ß
$batch_normalization_37/batchnorm/subSub9batch_normalization_37/batchnorm/ReadVariableOp_2:value:0*batch_normalization_37/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/subå
&batch_normalization_37/batchnorm/add_1AddV2*batch_normalization_37/batchnorm/mul_1:z:0(batch_normalization_37/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2(
&batch_normalization_37/batchnorm/add_1
max_pooling1d_37/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_37/ExpandDims/dimØ
max_pooling1d_37/ExpandDims
ExpandDims*batch_normalization_37/batchnorm/add_1:z:0(max_pooling1d_37/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
max_pooling1d_37/ExpandDimsÑ
max_pooling1d_37/MaxPoolMaxPool$max_pooling1d_37/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
max_pooling1d_37/MaxPool¯
max_pooling1d_37/SqueezeSqueeze!max_pooling1d_37/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
max_pooling1d_37/Squeezeu
flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿØ   2
flatten_18/Const¤
flatten_18/ReshapeReshape!max_pooling1d_37/Squeeze:output:0flatten_18/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
flatten_18/Reshape©
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02 
dense_54/MatMul/ReadVariableOp£
dense_54/MatMulMatMulflatten_18/Reshape:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_54/MatMul§
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_54/BiasAdd/ReadVariableOp¥
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_54/BiasAdds
dense_54/TanhTanhdense_54/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_54/Tanh{
dropout_36/IdentityIdentitydense_54/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_36/Identity¨
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_55/MatMul/ReadVariableOp¤
dense_55/MatMulMatMuldropout_36/Identity:output:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/MatMul§
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_55/BiasAdd/ReadVariableOp¥
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/BiasAdds
dense_55/TanhTanhdense_55/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/Tanh{
dropout_37/IdentityIdentitydense_55/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_37/Identity¨
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_56/MatMul/ReadVariableOp¤
dense_56/MatMulMatMuldropout_37/Identity:output:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_56/MatMul§
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_56/BiasAdd/ReadVariableOp¥
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_56/BiasAdd|
dense_56/SoftmaxSoftmaxdense_56/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_56/Softmaxû
IdentityIdentitydense_56/Softmax:softmax:00^batch_normalization_36/batchnorm/ReadVariableOp2^batch_normalization_36/batchnorm/ReadVariableOp_12^batch_normalization_36/batchnorm/ReadVariableOp_24^batch_normalization_36/batchnorm/mul/ReadVariableOp0^batch_normalization_37/batchnorm/ReadVariableOp2^batch_normalization_37/batchnorm/ReadVariableOp_12^batch_normalization_37/batchnorm/ReadVariableOp_24^batch_normalization_37/batchnorm/mul/ReadVariableOp!^conv1d_36/BiasAdd/ReadVariableOp-^conv1d_36/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_37/BiasAdd/ReadVariableOp-^conv1d_37/conv1d/ExpandDims_1/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::2b
/batch_normalization_36/batchnorm/ReadVariableOp/batch_normalization_36/batchnorm/ReadVariableOp2f
1batch_normalization_36/batchnorm/ReadVariableOp_11batch_normalization_36/batchnorm/ReadVariableOp_12f
1batch_normalization_36/batchnorm/ReadVariableOp_21batch_normalization_36/batchnorm/ReadVariableOp_22j
3batch_normalization_36/batchnorm/mul/ReadVariableOp3batch_normalization_36/batchnorm/mul/ReadVariableOp2b
/batch_normalization_37/batchnorm/ReadVariableOp/batch_normalization_37/batchnorm/ReadVariableOp2f
1batch_normalization_37/batchnorm/ReadVariableOp_11batch_normalization_37/batchnorm/ReadVariableOp_12f
1batch_normalization_37/batchnorm/ReadVariableOp_21batch_normalization_37/batchnorm/ReadVariableOp_22j
3batch_normalization_37/batchnorm/mul/ReadVariableOp3batch_normalization_37/batchnorm/mul/ReadVariableOp2D
 conv1d_36/BiasAdd/ReadVariableOp conv1d_36/BiasAdd/ReadVariableOp2\
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_37/BiasAdd/ReadVariableOp conv1d_37/BiasAdd/ReadVariableOp2\
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
ç	
Ý
D__inference_dense_54_layer_call_and_return_conditional_losses_399358

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
ð
ª
7__inference_batch_normalization_36_layer_call_fn_399052

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_3977322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô0
É
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_397732

inputs
assignmovingavg_397707
assignmovingavg_1_397713)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/397707*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_397707*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/397707*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/397707*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_397707AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/397707*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/397713*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_397713*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/397713*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/397713*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_397713AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/397713*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_397920

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

M
1__inference_max_pooling1d_37_layer_call_fn_397946

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_37_layer_call_and_return_conditional_losses_3979402
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
G
+__inference_flatten_18_layer_call_fn_399347

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_3982042
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

M
1__inference_max_pooling1d_36_layer_call_fn_397791

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_36_layer_call_and_return_conditional_losses_3977852
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ø
E__inference_conv1d_37_layer_call_and_return_conditional_losses_398090

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
½0
É
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399101

inputs
assignmovingavg_399076
assignmovingavg_1_399082)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/399076*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_399076*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/399076*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/399076*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_399076AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/399076*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/399082*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_399082*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399082*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/399082*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_399082AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/399082*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿH::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs

ø
E__inference_conv1d_36_layer_call_and_return_conditional_losses_397966

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1¶
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
Relu¨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿH::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
¨=
³
I__inference_sequential_18_layer_call_and_return_conditional_losses_398461

inputs
conv1d_36_398412
conv1d_36_398414!
batch_normalization_36_398417!
batch_normalization_36_398419!
batch_normalization_36_398421!
batch_normalization_36_398423
conv1d_37_398427
conv1d_37_398429!
batch_normalization_37_398432!
batch_normalization_37_398434!
batch_normalization_37_398436!
batch_normalization_37_398438
dense_54_398443
dense_54_398445
dense_55_398449
dense_55_398451
dense_56_398455
dense_56_398457
identity¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢!conv1d_36/StatefulPartitionedCall¢!conv1d_37/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall¢ dense_56/StatefulPartitionedCall¢"dropout_36/StatefulPartitionedCall¢"dropout_37/StatefulPartitionedCall¢
!conv1d_36/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_36_398412conv1d_36_398414*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_36_layer_call_and_return_conditional_losses_3979662#
!conv1d_36/StatefulPartitionedCallÇ
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv1d_36/StatefulPartitionedCall:output:0batch_normalization_36_398417batch_normalization_36_398419batch_normalization_36_398421batch_normalization_36_398423*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_39801720
.batch_normalization_36/StatefulPartitionedCall¦
 max_pooling1d_36/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_36_layer_call_and_return_conditional_losses_3977852"
 max_pooling1d_36/PartitionedCallÅ
!conv1d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_36/PartitionedCall:output:0conv1d_37_398427conv1d_37_398429*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_37_layer_call_and_return_conditional_losses_3980902#
!conv1d_37/StatefulPartitionedCallÇ
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv1d_37/StatefulPartitionedCall:output:0batch_normalization_37_398432batch_normalization_37_398434batch_normalization_37_398436batch_normalization_37_398438*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_39814120
.batch_normalization_37/StatefulPartitionedCall¦
 max_pooling1d_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_37_layer_call_and_return_conditional_losses_3979402"
 max_pooling1d_37/PartitionedCall
flatten_18/PartitionedCallPartitionedCall)max_pooling1d_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_3982042
flatten_18/PartitionedCall¶
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_54_398443dense_54_398445*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_3982232"
 dense_54/StatefulPartitionedCall
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_3982512$
"dropout_36/StatefulPartitionedCall¾
 dense_55/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0dense_55_398449dense_55_398451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_55_layer_call_and_return_conditional_losses_3982802"
 dense_55/StatefulPartitionedCall¿
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_3983082$
"dropout_37/StatefulPartitionedCall¾
 dense_56/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_56_398455dense_56_398457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_3983372"
 dense_56/StatefulPartitionedCallÚ
IdentityIdentity)dense_56/StatefulPartitionedCall:output:0/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall"^conv1d_36/StatefulPartitionedCall"^conv1d_37/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2F
!conv1d_36/StatefulPartitionedCall!conv1d_36/StatefulPartitionedCall2F
!conv1d_37/StatefulPartitionedCall!conv1d_37/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
Î
ª
7__inference_batch_normalization_36_layer_call_fn_399147

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_3980372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿH::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
½0
É
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_398141

inputs
assignmovingavg_398116
assignmovingavg_1_398122)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/398116*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_398116*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/398116*
_output_shapes
:2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/398116*
_output_shapes
:2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_398116AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/398116*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/398122*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_398122*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398122*
_output_shapes
:2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/398122*
_output_shapes
:2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_398122AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/398122*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ$::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
º:
ò
I__inference_sequential_18_layer_call_and_return_conditional_losses_398406
conv1d_36_input
conv1d_36_398357
conv1d_36_398359!
batch_normalization_36_398362!
batch_normalization_36_398364!
batch_normalization_36_398366!
batch_normalization_36_398368
conv1d_37_398372
conv1d_37_398374!
batch_normalization_37_398377!
batch_normalization_37_398379!
batch_normalization_37_398381!
batch_normalization_37_398383
dense_54_398388
dense_54_398390
dense_55_398394
dense_55_398396
dense_56_398400
dense_56_398402
identity¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢!conv1d_36/StatefulPartitionedCall¢!conv1d_37/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall¢ dense_56/StatefulPartitionedCall«
!conv1d_36/StatefulPartitionedCallStatefulPartitionedCallconv1d_36_inputconv1d_36_398357conv1d_36_398359*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_36_layer_call_and_return_conditional_losses_3979662#
!conv1d_36/StatefulPartitionedCallÉ
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv1d_36/StatefulPartitionedCall:output:0batch_normalization_36_398362batch_normalization_36_398364batch_normalization_36_398366batch_normalization_36_398368*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_39803720
.batch_normalization_36/StatefulPartitionedCall¦
 max_pooling1d_36/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_36_layer_call_and_return_conditional_losses_3977852"
 max_pooling1d_36/PartitionedCallÅ
!conv1d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_36/PartitionedCall:output:0conv1d_37_398372conv1d_37_398374*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_37_layer_call_and_return_conditional_losses_3980902#
!conv1d_37/StatefulPartitionedCallÉ
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv1d_37/StatefulPartitionedCall:output:0batch_normalization_37_398377batch_normalization_37_398379batch_normalization_37_398381batch_normalization_37_398383*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_39816120
.batch_normalization_37/StatefulPartitionedCall¦
 max_pooling1d_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_37_layer_call_and_return_conditional_losses_3979402"
 max_pooling1d_37/PartitionedCall
flatten_18/PartitionedCallPartitionedCall)max_pooling1d_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_3982042
flatten_18/PartitionedCall¶
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_54_398388dense_54_398390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_3982232"
 dense_54/StatefulPartitionedCall
dropout_36/PartitionedCallPartitionedCall)dense_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_3982562
dropout_36/PartitionedCall¶
 dense_55/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0dense_55_398394dense_55_398396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_55_layer_call_and_return_conditional_losses_3982802"
 dense_55/StatefulPartitionedCall
dropout_37/PartitionedCallPartitionedCall)dense_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_3983132
dropout_37/PartitionedCall¶
 dense_56/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_56_398400dense_56_398402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_3983372"
 dense_56/StatefulPartitionedCall
IdentityIdentity)dense_56/StatefulPartitionedCall:output:0/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall"^conv1d_36/StatefulPartitionedCall"^conv1d_37/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2F
!conv1d_36/StatefulPartitionedCall!conv1d_36/StatefulPartitionedCall2F
!conv1d_37/StatefulPartitionedCall!conv1d_37/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
)
_user_specified_nameconv1d_36_input
ã
~
)__inference_dense_54_layer_call_fn_399367

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_3982232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
é

R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399121

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿH::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs

e
F__inference_dropout_37_layer_call_and_return_conditional_losses_398308

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö	
Ý
D__inference_dense_56_layer_call_and_return_conditional_losses_399452

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
h
L__inference_max_pooling1d_37_layer_call_and_return_conditional_losses_397940

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDims°
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
ª
7__inference_batch_normalization_36_layer_call_fn_399134

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_3980172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿH::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
Ã=
¼
I__inference_sequential_18_layer_call_and_return_conditional_losses_398354
conv1d_36_input
conv1d_36_397977
conv1d_36_397979!
batch_normalization_36_398064!
batch_normalization_36_398066!
batch_normalization_36_398068!
batch_normalization_36_398070
conv1d_37_398101
conv1d_37_398103!
batch_normalization_37_398188!
batch_normalization_37_398190!
batch_normalization_37_398192!
batch_normalization_37_398194
dense_54_398234
dense_54_398236
dense_55_398291
dense_55_398293
dense_56_398348
dense_56_398350
identity¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢!conv1d_36/StatefulPartitionedCall¢!conv1d_37/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall¢ dense_56/StatefulPartitionedCall¢"dropout_36/StatefulPartitionedCall¢"dropout_37/StatefulPartitionedCall«
!conv1d_36/StatefulPartitionedCallStatefulPartitionedCallconv1d_36_inputconv1d_36_397977conv1d_36_397979*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_36_layer_call_and_return_conditional_losses_3979662#
!conv1d_36/StatefulPartitionedCallÇ
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv1d_36/StatefulPartitionedCall:output:0batch_normalization_36_398064batch_normalization_36_398066batch_normalization_36_398068batch_normalization_36_398070*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_39801720
.batch_normalization_36/StatefulPartitionedCall¦
 max_pooling1d_36/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_36_layer_call_and_return_conditional_losses_3977852"
 max_pooling1d_36/PartitionedCallÅ
!conv1d_37/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_36/PartitionedCall:output:0conv1d_37_398101conv1d_37_398103*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_37_layer_call_and_return_conditional_losses_3980902#
!conv1d_37/StatefulPartitionedCallÇ
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv1d_37/StatefulPartitionedCall:output:0batch_normalization_37_398188batch_normalization_37_398190batch_normalization_37_398192batch_normalization_37_398194*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_39814120
.batch_normalization_37/StatefulPartitionedCall¦
 max_pooling1d_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_37_layer_call_and_return_conditional_losses_3979402"
 max_pooling1d_37/PartitionedCall
flatten_18/PartitionedCallPartitionedCall)max_pooling1d_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_18_layer_call_and_return_conditional_losses_3982042
flatten_18/PartitionedCall¶
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_54_398234dense_54_398236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_3982232"
 dense_54/StatefulPartitionedCall
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_3982512$
"dropout_36/StatefulPartitionedCall¾
 dense_55/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0dense_55_398291dense_55_398293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_55_layer_call_and_return_conditional_losses_3982802"
 dense_55/StatefulPartitionedCall¿
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_3983082$
"dropout_37/StatefulPartitionedCall¾
 dense_56/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_56_398348dense_56_398350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_3983372"
 dense_56/StatefulPartitionedCallÚ
IdentityIdentity)dense_56/StatefulPartitionedCall:output:0/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall"^conv1d_36/StatefulPartitionedCall"^conv1d_37/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2F
!conv1d_36/StatefulPartitionedCall!conv1d_36/StatefulPartitionedCall2F
!conv1d_37/StatefulPartitionedCall!conv1d_37/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
)
_user_specified_nameconv1d_36_input
ò
ª
7__inference_batch_normalization_36_layer_call_fn_399065

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_3977652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²î
¶
I__inference_sequential_18_layer_call_and_return_conditional_losses_398783

inputs9
5conv1d_36_conv1d_expanddims_1_readvariableop_resource-
)conv1d_36_biasadd_readvariableop_resource1
-batch_normalization_36_assignmovingavg_3986673
/batch_normalization_36_assignmovingavg_1_398673@
<batch_normalization_36_batchnorm_mul_readvariableop_resource<
8batch_normalization_36_batchnorm_readvariableop_resource9
5conv1d_37_conv1d_expanddims_1_readvariableop_resource-
)conv1d_37_biasadd_readvariableop_resource1
-batch_normalization_37_assignmovingavg_3987153
/batch_normalization_37_assignmovingavg_1_398721@
<batch_normalization_37_batchnorm_mul_readvariableop_resource<
8batch_normalization_37_batchnorm_readvariableop_resource+
'dense_54_matmul_readvariableop_resource,
(dense_54_biasadd_readvariableop_resource+
'dense_55_matmul_readvariableop_resource,
(dense_55_biasadd_readvariableop_resource+
'dense_56_matmul_readvariableop_resource,
(dense_56_biasadd_readvariableop_resource
identity¢:batch_normalization_36/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_36/AssignMovingAvg/ReadVariableOp¢<batch_normalization_36/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_36/batchnorm/ReadVariableOp¢3batch_normalization_36/batchnorm/mul/ReadVariableOp¢:batch_normalization_37/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_37/AssignMovingAvg/ReadVariableOp¢<batch_normalization_37/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_37/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_37/batchnorm/ReadVariableOp¢3batch_normalization_37/batchnorm/mul/ReadVariableOp¢ conv1d_36/BiasAdd/ReadVariableOp¢,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp¢ conv1d_37/BiasAdd/ReadVariableOp¢,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp¢dense_54/BiasAdd/ReadVariableOp¢dense_54/MatMul/ReadVariableOp¢dense_55/BiasAdd/ReadVariableOp¢dense_55/MatMul/ReadVariableOp¢dense_56/BiasAdd/ReadVariableOp¢dense_56/MatMul/ReadVariableOp
conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_36/conv1d/ExpandDims/dim´
conv1d_36/conv1d/ExpandDims
ExpandDimsinputs(conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
conv1d_36/conv1d/ExpandDimsÖ
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_36_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_36/conv1d/ExpandDims_1/dimß
conv1d_36/conv1d/ExpandDims_1
ExpandDims4conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_36/conv1d/ExpandDims_1Þ
conv1d_36/conv1dConv2D$conv1d_36/conv1d/ExpandDims:output:0&conv1d_36/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
2
conv1d_36/conv1d°
conv1d_36/conv1d/SqueezeSqueezeconv1d_36/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_36/conv1d/Squeezeª
 conv1d_36/BiasAdd/ReadVariableOpReadVariableOp)conv1d_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_36/BiasAdd/ReadVariableOp´
conv1d_36/BiasAddBiasAdd!conv1d_36/conv1d/Squeeze:output:0(conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
conv1d_36/BiasAddz
conv1d_36/ReluReluconv1d_36/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
conv1d_36/Relu¿
5batch_normalization_36/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_36/moments/mean/reduction_indicesî
#batch_normalization_36/moments/meanMeanconv1d_36/Relu:activations:0>batch_normalization_36/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_36/moments/meanÅ
+batch_normalization_36/moments/StopGradientStopGradient,batch_normalization_36/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_36/moments/StopGradient
0batch_normalization_36/moments/SquaredDifferenceSquaredDifferenceconv1d_36/Relu:activations:04batch_normalization_36/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH22
0batch_normalization_36/moments/SquaredDifferenceÇ
9batch_normalization_36/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_36/moments/variance/reduction_indices
'batch_normalization_36/moments/varianceMean4batch_normalization_36/moments/SquaredDifference:z:0Bbatch_normalization_36/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_36/moments/varianceÆ
&batch_normalization_36/moments/SqueezeSqueeze,batch_normalization_36/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_36/moments/SqueezeÎ
(batch_normalization_36/moments/Squeeze_1Squeeze0batch_normalization_36/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_36/moments/Squeeze_1
,batch_normalization_36/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_36/AssignMovingAvg/398667*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_36/AssignMovingAvg/decayØ
5batch_normalization_36/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_36_assignmovingavg_398667*
_output_shapes
:*
dtype027
5batch_normalization_36/AssignMovingAvg/ReadVariableOpä
*batch_normalization_36/AssignMovingAvg/subSub=batch_normalization_36/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_36/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_36/AssignMovingAvg/398667*
_output_shapes
:2,
*batch_normalization_36/AssignMovingAvg/subÛ
*batch_normalization_36/AssignMovingAvg/mulMul.batch_normalization_36/AssignMovingAvg/sub:z:05batch_normalization_36/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_36/AssignMovingAvg/398667*
_output_shapes
:2,
*batch_normalization_36/AssignMovingAvg/mul¹
:batch_normalization_36/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_36_assignmovingavg_398667.batch_normalization_36/AssignMovingAvg/mul:z:06^batch_normalization_36/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_36/AssignMovingAvg/398667*
_output_shapes
 *
dtype02<
:batch_normalization_36/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_36/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_36/AssignMovingAvg_1/398673*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_36/AssignMovingAvg_1/decayÞ
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_36_assignmovingavg_1_398673*
_output_shapes
:*
dtype029
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOpî
,batch_normalization_36/AssignMovingAvg_1/subSub?batch_normalization_36/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_36/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_36/AssignMovingAvg_1/398673*
_output_shapes
:2.
,batch_normalization_36/AssignMovingAvg_1/subå
,batch_normalization_36/AssignMovingAvg_1/mulMul0batch_normalization_36/AssignMovingAvg_1/sub:z:07batch_normalization_36/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_36/AssignMovingAvg_1/398673*
_output_shapes
:2.
,batch_normalization_36/AssignMovingAvg_1/mulÅ
<batch_normalization_36/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_36_assignmovingavg_1_3986730batch_normalization_36/AssignMovingAvg_1/mul:z:08^batch_normalization_36/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_36/AssignMovingAvg_1/398673*
_output_shapes
 *
dtype02>
<batch_normalization_36/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_36/batchnorm/add/yÞ
$batch_normalization_36/batchnorm/addAddV21batch_normalization_36/moments/Squeeze_1:output:0/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_36/batchnorm/add¨
&batch_normalization_36/batchnorm/RsqrtRsqrt(batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_36/batchnorm/Rsqrtã
3batch_normalization_36/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_36_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_36/batchnorm/mul/ReadVariableOpá
$batch_normalization_36/batchnorm/mulMul*batch_normalization_36/batchnorm/Rsqrt:y:0;batch_normalization_36/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_36/batchnorm/mulÕ
&batch_normalization_36/batchnorm/mul_1Mulconv1d_36/Relu:activations:0(batch_normalization_36/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2(
&batch_normalization_36/batchnorm/mul_1×
&batch_normalization_36/batchnorm/mul_2Mul/batch_normalization_36/moments/Squeeze:output:0(batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_36/batchnorm/mul_2×
/batch_normalization_36/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_36_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_36/batchnorm/ReadVariableOpÝ
$batch_normalization_36/batchnorm/subSub7batch_normalization_36/batchnorm/ReadVariableOp:value:0*batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_36/batchnorm/subå
&batch_normalization_36/batchnorm/add_1AddV2*batch_normalization_36/batchnorm/mul_1:z:0(batch_normalization_36/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2(
&batch_normalization_36/batchnorm/add_1
max_pooling1d_36/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_36/ExpandDims/dimØ
max_pooling1d_36/ExpandDims
ExpandDims*batch_normalization_36/batchnorm/add_1:z:0(max_pooling1d_36/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH2
max_pooling1d_36/ExpandDimsÑ
max_pooling1d_36/MaxPoolMaxPool$max_pooling1d_36/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
ksize
*
paddingSAME*
strides
2
max_pooling1d_36/MaxPool¯
max_pooling1d_36/SqueezeSqueeze!max_pooling1d_36/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
squeeze_dims
2
max_pooling1d_36/Squeeze
conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_37/conv1d/ExpandDims/dimÏ
conv1d_37/conv1d/ExpandDims
ExpandDims!max_pooling1d_36/Squeeze:output:0(conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
conv1d_37/conv1d/ExpandDimsÖ
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_37/conv1d/ExpandDims_1/dimß
conv1d_37/conv1d/ExpandDims_1
ExpandDims4conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_37/conv1d/ExpandDims_1Þ
conv1d_37/conv1dConv2D$conv1d_37/conv1d/ExpandDims:output:0&conv1d_37/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
paddingSAME*
strides
2
conv1d_37/conv1d°
conv1d_37/conv1d/SqueezeSqueezeconv1d_37/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_37/conv1d/Squeezeª
 conv1d_37/BiasAdd/ReadVariableOpReadVariableOp)conv1d_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_37/BiasAdd/ReadVariableOp´
conv1d_37/BiasAddBiasAdd!conv1d_37/conv1d/Squeeze:output:0(conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
conv1d_37/BiasAddz
conv1d_37/ReluReluconv1d_37/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
conv1d_37/Relu¿
5batch_normalization_37/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_37/moments/mean/reduction_indicesî
#batch_normalization_37/moments/meanMeanconv1d_37/Relu:activations:0>batch_normalization_37/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_37/moments/meanÅ
+batch_normalization_37/moments/StopGradientStopGradient,batch_normalization_37/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_37/moments/StopGradient
0batch_normalization_37/moments/SquaredDifferenceSquaredDifferenceconv1d_37/Relu:activations:04batch_normalization_37/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$22
0batch_normalization_37/moments/SquaredDifferenceÇ
9batch_normalization_37/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_37/moments/variance/reduction_indices
'batch_normalization_37/moments/varianceMean4batch_normalization_37/moments/SquaredDifference:z:0Bbatch_normalization_37/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_37/moments/varianceÆ
&batch_normalization_37/moments/SqueezeSqueeze,batch_normalization_37/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_37/moments/SqueezeÎ
(batch_normalization_37/moments/Squeeze_1Squeeze0batch_normalization_37/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_37/moments/Squeeze_1
,batch_normalization_37/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_37/AssignMovingAvg/398715*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_37/AssignMovingAvg/decayØ
5batch_normalization_37/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_37_assignmovingavg_398715*
_output_shapes
:*
dtype027
5batch_normalization_37/AssignMovingAvg/ReadVariableOpä
*batch_normalization_37/AssignMovingAvg/subSub=batch_normalization_37/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_37/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_37/AssignMovingAvg/398715*
_output_shapes
:2,
*batch_normalization_37/AssignMovingAvg/subÛ
*batch_normalization_37/AssignMovingAvg/mulMul.batch_normalization_37/AssignMovingAvg/sub:z:05batch_normalization_37/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_37/AssignMovingAvg/398715*
_output_shapes
:2,
*batch_normalization_37/AssignMovingAvg/mul¹
:batch_normalization_37/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_37_assignmovingavg_398715.batch_normalization_37/AssignMovingAvg/mul:z:06^batch_normalization_37/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_37/AssignMovingAvg/398715*
_output_shapes
 *
dtype02<
:batch_normalization_37/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_37/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_37/AssignMovingAvg_1/398721*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_37/AssignMovingAvg_1/decayÞ
7batch_normalization_37/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_37_assignmovingavg_1_398721*
_output_shapes
:*
dtype029
7batch_normalization_37/AssignMovingAvg_1/ReadVariableOpî
,batch_normalization_37/AssignMovingAvg_1/subSub?batch_normalization_37/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_37/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_37/AssignMovingAvg_1/398721*
_output_shapes
:2.
,batch_normalization_37/AssignMovingAvg_1/subå
,batch_normalization_37/AssignMovingAvg_1/mulMul0batch_normalization_37/AssignMovingAvg_1/sub:z:07batch_normalization_37/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_37/AssignMovingAvg_1/398721*
_output_shapes
:2.
,batch_normalization_37/AssignMovingAvg_1/mulÅ
<batch_normalization_37/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_37_assignmovingavg_1_3987210batch_normalization_37/AssignMovingAvg_1/mul:z:08^batch_normalization_37/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_37/AssignMovingAvg_1/398721*
_output_shapes
 *
dtype02>
<batch_normalization_37/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_37/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_37/batchnorm/add/yÞ
$batch_normalization_37/batchnorm/addAddV21batch_normalization_37/moments/Squeeze_1:output:0/batch_normalization_37/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/add¨
&batch_normalization_37/batchnorm/RsqrtRsqrt(batch_normalization_37/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/Rsqrtã
3batch_normalization_37/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_37_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_37/batchnorm/mul/ReadVariableOpá
$batch_normalization_37/batchnorm/mulMul*batch_normalization_37/batchnorm/Rsqrt:y:0;batch_normalization_37/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/mulÕ
&batch_normalization_37/batchnorm/mul_1Mulconv1d_37/Relu:activations:0(batch_normalization_37/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2(
&batch_normalization_37/batchnorm/mul_1×
&batch_normalization_37/batchnorm/mul_2Mul/batch_normalization_37/moments/Squeeze:output:0(batch_normalization_37/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/mul_2×
/batch_normalization_37/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_37_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_37/batchnorm/ReadVariableOpÝ
$batch_normalization_37/batchnorm/subSub7batch_normalization_37/batchnorm/ReadVariableOp:value:0*batch_normalization_37/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/subå
&batch_normalization_37/batchnorm/add_1AddV2*batch_normalization_37/batchnorm/mul_1:z:0(batch_normalization_37/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2(
&batch_normalization_37/batchnorm/add_1
max_pooling1d_37/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_37/ExpandDims/dimØ
max_pooling1d_37/ExpandDims
ExpandDims*batch_normalization_37/batchnorm/add_1:z:0(max_pooling1d_37/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2
max_pooling1d_37/ExpandDimsÑ
max_pooling1d_37/MaxPoolMaxPool$max_pooling1d_37/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
max_pooling1d_37/MaxPool¯
max_pooling1d_37/SqueezeSqueeze!max_pooling1d_37/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
max_pooling1d_37/Squeezeu
flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿØ   2
flatten_18/Const¤
flatten_18/ReshapeReshape!max_pooling1d_37/Squeeze:output:0flatten_18/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
flatten_18/Reshape©
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes
:	Ø*
dtype02 
dense_54/MatMul/ReadVariableOp£
dense_54/MatMulMatMulflatten_18/Reshape:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_54/MatMul§
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_54/BiasAdd/ReadVariableOp¥
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_54/BiasAdds
dense_54/TanhTanhdense_54/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_54/Tanhy
dropout_36/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout_36/dropout/Const
dropout_36/dropout/MulMuldense_54/Tanh:y:0!dropout_36/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_36/dropout/Mulu
dropout_36/dropout/ShapeShapedense_54/Tanh:y:0*
T0*
_output_shapes
:2
dropout_36/dropout/ShapeÕ
/dropout_36/dropout/random_uniform/RandomUniformRandomUniform!dropout_36/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_36/dropout/random_uniform/RandomUniform
!dropout_36/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2#
!dropout_36/dropout/GreaterEqual/yê
dropout_36/dropout/GreaterEqualGreaterEqual8dropout_36/dropout/random_uniform/RandomUniform:output:0*dropout_36/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_36/dropout/GreaterEqual 
dropout_36/dropout/CastCast#dropout_36/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_36/dropout/Cast¦
dropout_36/dropout/Mul_1Muldropout_36/dropout/Mul:z:0dropout_36/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_36/dropout/Mul_1¨
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_55/MatMul/ReadVariableOp¤
dense_55/MatMulMatMuldropout_36/dropout/Mul_1:z:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/MatMul§
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_55/BiasAdd/ReadVariableOp¥
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/BiasAdds
dense_55/TanhTanhdense_55/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/Tanhy
dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout_37/dropout/Const
dropout_37/dropout/MulMuldense_55/Tanh:y:0!dropout_37/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_37/dropout/Mulu
dropout_37/dropout/ShapeShapedense_55/Tanh:y:0*
T0*
_output_shapes
:2
dropout_37/dropout/ShapeÕ
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_37/dropout/random_uniform/RandomUniform
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2#
!dropout_37/dropout/GreaterEqual/yê
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_37/dropout/GreaterEqual 
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_37/dropout/Cast¦
dropout_37/dropout/Mul_1Muldropout_37/dropout/Mul:z:0dropout_37/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_37/dropout/Mul_1¨
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_56/MatMul/ReadVariableOp¤
dense_56/MatMulMatMuldropout_37/dropout/Mul_1:z:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_56/MatMul§
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_56/BiasAdd/ReadVariableOp¥
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_56/BiasAdd|
dense_56/SoftmaxSoftmaxdense_56/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_56/Softmax	
IdentityIdentitydense_56/Softmax:softmax:0;^batch_normalization_36/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_36/AssignMovingAvg/ReadVariableOp=^batch_normalization_36/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_36/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_36/batchnorm/ReadVariableOp4^batch_normalization_36/batchnorm/mul/ReadVariableOp;^batch_normalization_37/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_37/AssignMovingAvg/ReadVariableOp=^batch_normalization_37/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_37/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_37/batchnorm/ReadVariableOp4^batch_normalization_37/batchnorm/mul/ReadVariableOp!^conv1d_36/BiasAdd/ReadVariableOp-^conv1d_36/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_37/BiasAdd/ReadVariableOp-^conv1d_37/conv1d/ExpandDims_1/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:ÿÿÿÿÿÿÿÿÿH::::::::::::::::::2x
:batch_normalization_36/AssignMovingAvg/AssignSubVariableOp:batch_normalization_36/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_36/AssignMovingAvg/ReadVariableOp5batch_normalization_36/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_36/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_36/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_36/batchnorm/ReadVariableOp/batch_normalization_36/batchnorm/ReadVariableOp2j
3batch_normalization_36/batchnorm/mul/ReadVariableOp3batch_normalization_36/batchnorm/mul/ReadVariableOp2x
:batch_normalization_37/AssignMovingAvg/AssignSubVariableOp:batch_normalization_37/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_37/AssignMovingAvg/ReadVariableOp5batch_normalization_37/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_37/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_37/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_37/AssignMovingAvg_1/ReadVariableOp7batch_normalization_37/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_37/batchnorm/ReadVariableOp/batch_normalization_37/batchnorm/ReadVariableOp2j
3batch_normalization_37/batchnorm/mul/ReadVariableOp3batch_normalization_37/batchnorm/mul/ReadVariableOp2D
 conv1d_36/BiasAdd/ReadVariableOp conv1d_36/BiasAdd/ReadVariableOp2\
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_37/BiasAdd/ReadVariableOp conv1d_37/BiasAdd/ReadVariableOp2\
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs


R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_397765

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
b
F__inference_flatten_18_layer_call_and_return_conditional_losses_398204

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿØ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
~
)__inference_dense_55_layer_call_fn_399414

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_55_layer_call_and_return_conditional_losses_3982802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
ª
7__inference_batch_normalization_37_layer_call_fn_399241

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_3981412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ$::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
¶
b
F__inference_flatten_18_layer_call_and_return_conditional_losses_399342

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿØ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
O
conv1d_36_input<
!serving_default_conv1d_36_input:0ÿÿÿÿÿÿÿÿÿH<
dense_560
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:î
ÖZ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
Ä_default_save_signature
Å__call__
+Æ&call_and_return_all_conditional_losses"´V
_tf_keras_sequentialV{"class_name": "Sequential", "name": "sequential_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_36_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_36", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_36", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_36", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_37", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_37", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_37", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_36_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_36", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_36", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_36", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_37", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_37", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_37", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
à


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"¹	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_36", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 1]}}
º	
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
 	variables
!	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization_36", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_36", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 12]}}
ü
"trainable_variables
#regularization_losses
$	variables
%	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "MaxPooling1D", "name": "max_pooling1d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_36", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
é	

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_37", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 12]}}
º	
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1trainable_variables
2regularization_losses
3	variables
4	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization_37", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_37", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 12]}}
ü
5trainable_variables
6regularization_losses
7	variables
8	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "MaxPooling1D", "name": "max_pooling1d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_37", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ê
9trainable_variables
:regularization_losses
;	variables
<	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
õ

=kernel
>bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 216}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 216]}}
é
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_36", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_36", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
ñ

Gkernel
Hbias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "dense_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
é
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_37", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_37", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
ô

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_56", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
ë

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate
[iterm¨m©mªm«&m¬'m­-m®.m¯=m°>m±Gm²Hm³Qm´Rmµv¶v·v¸v¹&vº'v»-v¼.v½=v¾>v¿GvÀHvÁQvÂRvÃ"
	optimizer
 "
trackable_list_wrapper
¦
0
1
2
3
4
5
&6
'7
-8
.9
/10
011
=12
>13
G14
H15
Q16
R17"
trackable_list_wrapper

0
1
2
3
&4
'5
-6
.7
=8
>9
G10
H11
Q12
R13"
trackable_list_wrapper
Î
\layer_regularization_losses
regularization_losses

]layers
^non_trainable_variables
	variables
_layer_metrics
trainable_variables
`metrics
Å__call__
Ä_default_save_signature
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
-
ßserving_default"
signature_map
&:$2conv1d_36/kernel
:2conv1d_36/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
trainable_variables
alayer_regularization_losses
regularization_losses

blayers
	variables
clayer_metrics
dnon_trainable_variables
emetrics
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_36/gamma
):'2batch_normalization_36/beta
2:0 (2"batch_normalization_36/moving_mean
6:4 (2&batch_normalization_36/moving_variance
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
°
trainable_variables
flayer_regularization_losses
regularization_losses

glayers
 	variables
hlayer_metrics
inon_trainable_variables
jmetrics
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
"trainable_variables
klayer_regularization_losses
#regularization_losses

llayers
$	variables
mlayer_metrics
nnon_trainable_variables
ometrics
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
&:$2conv1d_37/kernel
:2conv1d_37/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
°
(trainable_variables
player_regularization_losses
)regularization_losses

qlayers
*	variables
rlayer_metrics
snon_trainable_variables
tmetrics
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_37/gamma
):'2batch_normalization_37/beta
2:0 (2"batch_normalization_37/moving_mean
6:4 (2&batch_normalization_37/moving_variance
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
°
1trainable_variables
ulayer_regularization_losses
2regularization_losses

vlayers
3	variables
wlayer_metrics
xnon_trainable_variables
ymetrics
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
5trainable_variables
zlayer_regularization_losses
6regularization_losses

{layers
7	variables
|layer_metrics
}non_trainable_variables
~metrics
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
´
9trainable_variables
layer_regularization_losses
:regularization_losses
layers
;	variables
layer_metrics
non_trainable_variables
metrics
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
": 	Ø2dense_54/kernel
:2dense_54/bias
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
µ
?trainable_variables
 layer_regularization_losses
@regularization_losses
layers
A	variables
layer_metrics
non_trainable_variables
metrics
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ctrainable_variables
 layer_regularization_losses
Dregularization_losses
layers
E	variables
layer_metrics
non_trainable_variables
metrics
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
!:2dense_55/kernel
:2dense_55/bias
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
µ
Itrainable_variables
 layer_regularization_losses
Jregularization_losses
layers
K	variables
layer_metrics
non_trainable_variables
metrics
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Mtrainable_variables
 layer_regularization_losses
Nregularization_losses
layers
O	variables
layer_metrics
non_trainable_variables
metrics
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
!:2dense_56/kernel
:2dense_56/bias
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
µ
Strainable_variables
 layer_regularization_losses
Tregularization_losses
layers
U	variables
layer_metrics
non_trainable_variables
metrics
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
 "
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
<
0
1
/2
03"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

total

 count
¡	variables
¢	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


£total

¤count
¥
_fn_kwargs
¦	variables
§	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
0
 1"
trackable_list_wrapper
.
¡	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
£0
¤1"
trackable_list_wrapper
.
¦	variables"
_generic_user_object
+:)2Adam/conv1d_36/kernel/m
!:2Adam/conv1d_36/bias/m
/:-2#Adam/batch_normalization_36/gamma/m
.:,2"Adam/batch_normalization_36/beta/m
+:)2Adam/conv1d_37/kernel/m
!:2Adam/conv1d_37/bias/m
/:-2#Adam/batch_normalization_37/gamma/m
.:,2"Adam/batch_normalization_37/beta/m
':%	Ø2Adam/dense_54/kernel/m
 :2Adam/dense_54/bias/m
&:$2Adam/dense_55/kernel/m
 :2Adam/dense_55/bias/m
&:$2Adam/dense_56/kernel/m
 :2Adam/dense_56/bias/m
+:)2Adam/conv1d_36/kernel/v
!:2Adam/conv1d_36/bias/v
/:-2#Adam/batch_normalization_36/gamma/v
.:,2"Adam/batch_normalization_36/beta/v
+:)2Adam/conv1d_37/kernel/v
!:2Adam/conv1d_37/bias/v
/:-2#Adam/batch_normalization_37/gamma/v
.:,2"Adam/batch_normalization_37/beta/v
':%	Ø2Adam/dense_54/kernel/v
 :2Adam/dense_54/bias/v
&:$2Adam/dense_55/kernel/v
 :2Adam/dense_55/bias/v
&:$2Adam/dense_56/kernel/v
 :2Adam/dense_56/bias/v
ë2è
!__inference__wrapped_model_397636Â
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *2¢/
-*
conv1d_36_inputÿÿÿÿÿÿÿÿÿH
2
.__inference_sequential_18_layer_call_fn_398917
.__inference_sequential_18_layer_call_fn_398958
.__inference_sequential_18_layer_call_fn_398500
.__inference_sequential_18_layer_call_fn_398593À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_18_layer_call_and_return_conditional_losses_398783
I__inference_sequential_18_layer_call_and_return_conditional_losses_398876
I__inference_sequential_18_layer_call_and_return_conditional_losses_398354
I__inference_sequential_18_layer_call_and_return_conditional_losses_398406À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
*__inference_conv1d_36_layer_call_fn_398983¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv1d_36_layer_call_and_return_conditional_losses_398974¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
7__inference_batch_normalization_36_layer_call_fn_399147
7__inference_batch_normalization_36_layer_call_fn_399052
7__inference_batch_normalization_36_layer_call_fn_399065
7__inference_batch_normalization_36_layer_call_fn_399134´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399101
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399019
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399039
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399121´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
1__inference_max_pooling1d_36_layer_call_fn_397791Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
§2¤
L__inference_max_pooling1d_36_layer_call_and_return_conditional_losses_397785Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_conv1d_37_layer_call_fn_399172¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv1d_37_layer_call_and_return_conditional_losses_399163¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
7__inference_batch_normalization_37_layer_call_fn_399323
7__inference_batch_normalization_37_layer_call_fn_399254
7__inference_batch_normalization_37_layer_call_fn_399241
7__inference_batch_normalization_37_layer_call_fn_399336´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399290
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399228
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399310
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399208´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
1__inference_max_pooling1d_37_layer_call_fn_397946Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
§2¤
L__inference_max_pooling1d_37_layer_call_and_return_conditional_losses_397940Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_flatten_18_layer_call_fn_399347¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_18_layer_call_and_return_conditional_losses_399342¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_54_layer_call_fn_399367¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_54_layer_call_and_return_conditional_losses_399358¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_dropout_36_layer_call_fn_399389
+__inference_dropout_36_layer_call_fn_399394´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_36_layer_call_and_return_conditional_losses_399379
F__inference_dropout_36_layer_call_and_return_conditional_losses_399384´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_dense_55_layer_call_fn_399414¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_55_layer_call_and_return_conditional_losses_399405¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_dropout_37_layer_call_fn_399436
+__inference_dropout_37_layer_call_fn_399441´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_37_layer_call_and_return_conditional_losses_399431
F__inference_dropout_37_layer_call_and_return_conditional_losses_399426´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_dense_56_layer_call_fn_399461¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_56_layer_call_and_return_conditional_losses_399452¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÓBÐ
$__inference_signature_wrapper_398644conv1d_36_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ­
!__inference__wrapped_model_397636&'0-/.=>GHQR<¢9
2¢/
-*
conv1d_36_inputÿÿÿÿÿÿÿÿÿH
ª "3ª0
.
dense_56"
dense_56ÿÿÿÿÿÿÿÿÿÒ
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399019|@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399039|@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399101j7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿH
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿH
 À
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_399121j7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿH
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿH
 ª
7__inference_batch_normalization_36_layer_call_fn_399052o@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
7__inference_batch_normalization_36_layer_call_fn_399065o@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_36_layer_call_fn_399134]7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿH
p
ª "ÿÿÿÿÿÿÿÿÿH
7__inference_batch_normalization_36_layer_call_fn_399147]7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿH
p 
ª "ÿÿÿÿÿÿÿÿÿHÀ
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399208j/0-.7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ$
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ$
 À
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399228j0-/.7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ$
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ$
 Ò
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399290|/0-.@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_399310|0-/.@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_37_layer_call_fn_399241]/0-.7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ$
p
ª "ÿÿÿÿÿÿÿÿÿ$
7__inference_batch_normalization_37_layer_call_fn_399254]0-/.7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ$
p 
ª "ÿÿÿÿÿÿÿÿÿ$ª
7__inference_batch_normalization_37_layer_call_fn_399323o/0-.@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
7__inference_batch_normalization_37_layer_call_fn_399336o0-/.@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ­
E__inference_conv1d_36_layer_call_and_return_conditional_losses_398974d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿH
ª ")¢&

0ÿÿÿÿÿÿÿÿÿH
 
*__inference_conv1d_36_layer_call_fn_398983W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿH
ª "ÿÿÿÿÿÿÿÿÿH­
E__inference_conv1d_37_layer_call_and_return_conditional_losses_399163d&'3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ$
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ$
 
*__inference_conv1d_37_layer_call_fn_399172W&'3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ$
ª "ÿÿÿÿÿÿÿÿÿ$¥
D__inference_dense_54_layer_call_and_return_conditional_losses_399358]=>0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿØ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense_54_layer_call_fn_399367P=>0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿØ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_55_layer_call_and_return_conditional_losses_399405\GH/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_55_layer_call_fn_399414OGH/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_56_layer_call_and_return_conditional_losses_399452\QR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_56_layer_call_fn_399461OQR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_36_layer_call_and_return_conditional_losses_399379\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
F__inference_dropout_36_layer_call_and_return_conditional_losses_399384\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dropout_36_layer_call_fn_399389O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ~
+__inference_dropout_36_layer_call_fn_399394O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_37_layer_call_and_return_conditional_losses_399426\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
F__inference_dropout_37_layer_call_and_return_conditional_losses_399431\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dropout_37_layer_call_fn_399436O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ~
+__inference_dropout_37_layer_call_fn_399441O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_flatten_18_layer_call_and_return_conditional_losses_399342]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿØ
 
+__inference_flatten_18_layer_call_fn_399347P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿØÕ
L__inference_max_pooling1d_36_layer_call_and_return_conditional_losses_397785E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¬
1__inference_max_pooling1d_36_layer_call_fn_397791wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÕ
L__inference_max_pooling1d_37_layer_call_and_return_conditional_losses_397940E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¬
1__inference_max_pooling1d_37_layer_call_fn_397946wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÏ
I__inference_sequential_18_layer_call_and_return_conditional_losses_398354&'/0-.=>GHQRD¢A
:¢7
-*
conv1d_36_inputÿÿÿÿÿÿÿÿÿH
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
I__inference_sequential_18_layer_call_and_return_conditional_losses_398406&'0-/.=>GHQRD¢A
:¢7
-*
conv1d_36_inputÿÿÿÿÿÿÿÿÿH
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
I__inference_sequential_18_layer_call_and_return_conditional_losses_398783x&'/0-.=>GHQR;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿH
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
I__inference_sequential_18_layer_call_and_return_conditional_losses_398876x&'0-/.=>GHQR;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿH
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
.__inference_sequential_18_layer_call_fn_398500t&'/0-.=>GHQRD¢A
:¢7
-*
conv1d_36_inputÿÿÿÿÿÿÿÿÿH
p

 
ª "ÿÿÿÿÿÿÿÿÿ¦
.__inference_sequential_18_layer_call_fn_398593t&'0-/.=>GHQRD¢A
:¢7
-*
conv1d_36_inputÿÿÿÿÿÿÿÿÿH
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_18_layer_call_fn_398917k&'/0-.=>GHQR;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿH
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_18_layer_call_fn_398958k&'0-/.=>GHQR;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿH
p 

 
ª "ÿÿÿÿÿÿÿÿÿÃ
$__inference_signature_wrapper_398644&'0-/.=>GHQRO¢L
¢ 
EªB
@
conv1d_36_input-*
conv1d_36_inputÿÿÿÿÿÿÿÿÿH"3ª0
.
dense_56"
dense_56ÿÿÿÿÿÿÿÿÿ