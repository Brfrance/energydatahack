оа
Б
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
О
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8УЇ

conv1d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_38/kernel
y
$conv1d_38/kernel/Read/ReadVariableOpReadVariableOpconv1d_38/kernel*"
_output_shapes
:*
dtype0
t
conv1d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_38/bias
m
"conv1d_38/bias/Read/ReadVariableOpReadVariableOpconv1d_38/bias*
_output_shapes
:*
dtype0

batch_normalization_38/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_38/gamma

0batch_normalization_38/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_38/gamma*
_output_shapes
:*
dtype0

batch_normalization_38/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_38/beta

/batch_normalization_38/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_38/beta*
_output_shapes
:*
dtype0

"batch_normalization_38/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_38/moving_mean

6batch_normalization_38/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_38/moving_mean*
_output_shapes
:*
dtype0
Є
&batch_normalization_38/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_38/moving_variance

:batch_normalization_38/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_38/moving_variance*
_output_shapes
:*
dtype0

conv1d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_39/kernel
y
$conv1d_39/kernel/Read/ReadVariableOpReadVariableOpconv1d_39/kernel*"
_output_shapes
:*
dtype0
t
conv1d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_39/bias
m
"conv1d_39/bias/Read/ReadVariableOpReadVariableOpconv1d_39/bias*
_output_shapes
:*
dtype0

batch_normalization_39/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_39/gamma

0batch_normalization_39/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_39/gamma*
_output_shapes
:*
dtype0

batch_normalization_39/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_39/beta

/batch_normalization_39/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_39/beta*
_output_shapes
:*
dtype0

"batch_normalization_39/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_39/moving_mean

6batch_normalization_39/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_39/moving_mean*
_output_shapes
:*
dtype0
Є
&batch_normalization_39/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_39/moving_variance

:batch_normalization_39/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_39/moving_variance*
_output_shapes
:*
dtype0
{
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и* 
shared_namedense_57/kernel
t
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes
:	и*
dtype0
r
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes
:*
dtype0
z
dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_58/kernel
s
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel*
_output_shapes

:*
dtype0
r
dense_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_58/bias
k
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes
:*
dtype0
z
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_59/kernel
s
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes

:*
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
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
Adam/conv1d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_38/kernel/m

+Adam/conv1d_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/kernel/m*"
_output_shapes
:*
dtype0

Adam/conv1d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_38/bias/m
{
)Adam/conv1d_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_38/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_38/gamma/m

7Adam/batch_normalization_38/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_38/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_38/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_38/beta/m

6Adam/batch_normalization_38/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_38/beta/m*
_output_shapes
:*
dtype0

Adam/conv1d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_39/kernel/m

+Adam/conv1d_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/kernel/m*"
_output_shapes
:*
dtype0

Adam/conv1d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_39/bias/m
{
)Adam/conv1d_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_39/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_39/gamma/m

7Adam/batch_normalization_39/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_39/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_39/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_39/beta/m

6Adam/batch_normalization_39/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_39/beta/m*
_output_shapes
:*
dtype0

Adam/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и*'
shared_nameAdam/dense_57/kernel/m

*Adam/dense_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/m*
_output_shapes
:	и*
dtype0

Adam/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_57/bias/m
y
(Adam/dense_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/m*
_output_shapes
:*
dtype0

Adam/dense_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_58/kernel/m

*Adam/dense_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_58/bias/m
y
(Adam/dense_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/m*
_output_shapes
:*
dtype0

Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_59/kernel/m

*Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_59/bias/m
y
(Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/m*
_output_shapes
:*
dtype0

Adam/conv1d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_38/kernel/v

+Adam/conv1d_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/kernel/v*"
_output_shapes
:*
dtype0

Adam/conv1d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_38/bias/v
{
)Adam/conv1d_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_38/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_38/gamma/v

7Adam/batch_normalization_38/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_38/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_38/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_38/beta/v

6Adam/batch_normalization_38/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_38/beta/v*
_output_shapes
:*
dtype0

Adam/conv1d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_39/kernel/v

+Adam/conv1d_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/kernel/v*"
_output_shapes
:*
dtype0

Adam/conv1d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_39/bias/v
{
)Adam/conv1d_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_39/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_39/gamma/v

7Adam/batch_normalization_39/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_39/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_39/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_39/beta/v

6Adam/batch_normalization_39/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_39/beta/v*
_output_shapes
:*
dtype0

Adam/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и*'
shared_nameAdam/dense_57/kernel/v

*Adam/dense_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/v*
_output_shapes
:	и*
dtype0

Adam/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_57/bias/v
y
(Adam/dense_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/v*
_output_shapes
:*
dtype0

Adam/dense_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_58/kernel/v

*Adam/dense_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_58/bias/v
y
(Adam/dense_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/v*
_output_shapes
:*
dtype0

Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_59/kernel/v

*Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_59/bias/v
y
(Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ѓ[
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*оZ
valueдZBбZ BЪZ
Х
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
и

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate
[itermЈmЉmЊmЋ&mЌ'm­-mЎ.mЏ=mА>mБGmВHmГQmДRmЕvЖvЗvИvЙ&vК'vЛ-vМ.vН=vО>vПGvРHvСQvТRvУ
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
VARIABLE_VALUEconv1d_38/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_38/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_38/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_38/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_38/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_38/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv1d_39/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_39/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_39/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_39/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_39/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_39/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
Б
9trainable_variables
layer_regularization_losses
:regularization_losses
layers
;	variables
layer_metrics
non_trainable_variables
metrics
[Y
VARIABLE_VALUEdense_57/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_57/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
 

=0
>1
В
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
В
Ctrainable_variables
 layer_regularization_losses
Dregularization_losses
layers
E	variables
layer_metrics
non_trainable_variables
metrics
[Y
VARIABLE_VALUEdense_58/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_58/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
 

G0
H1
В
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
В
Mtrainable_variables
 layer_regularization_losses
Nregularization_losses
layers
O	variables
layer_metrics
non_trainable_variables
metrics
[Y
VARIABLE_VALUEdense_59/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_59/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
 

Q0
R1
В
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
Ё	variables
Ђ	keras_api
I

Ѓtotal

Єcount
Ѕ
_fn_kwargs
І	variables
Ї	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
 1

Ё	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ѓ0
Є1

І	variables
}
VARIABLE_VALUEAdam/conv1d_38/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_38/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_38/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_38/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_39/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_39/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_39/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_39/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_57/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_57/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_58/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_58/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_59/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_59/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_38/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_38/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_38/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_38/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_39/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_39/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_39/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_39/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_57/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_57/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_58/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_58/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_59/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_59/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv1d_38_inputPlaceholder*+
_output_shapes
:џџџџџџџџџH*
dtype0* 
shape:џџџџџџџџџH

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_38_inputconv1d_38/kernelconv1d_38/bias&batch_normalization_38/moving_variancebatch_normalization_38/gamma"batch_normalization_38/moving_meanbatch_normalization_38/betaconv1d_39/kernelconv1d_39/bias&batch_normalization_39/moving_variancebatch_normalization_39/gamma"batch_normalization_39/moving_meanbatch_normalization_39/betadense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_419847
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_38/kernel/Read/ReadVariableOp"conv1d_38/bias/Read/ReadVariableOp0batch_normalization_38/gamma/Read/ReadVariableOp/batch_normalization_38/beta/Read/ReadVariableOp6batch_normalization_38/moving_mean/Read/ReadVariableOp:batch_normalization_38/moving_variance/Read/ReadVariableOp$conv1d_39/kernel/Read/ReadVariableOp"conv1d_39/bias/Read/ReadVariableOp0batch_normalization_39/gamma/Read/ReadVariableOp/batch_normalization_39/beta/Read/ReadVariableOp6batch_normalization_39/moving_mean/Read/ReadVariableOp:batch_normalization_39/moving_variance/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOp#dense_58/kernel/Read/ReadVariableOp!dense_58/bias/Read/ReadVariableOp#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv1d_38/kernel/m/Read/ReadVariableOp)Adam/conv1d_38/bias/m/Read/ReadVariableOp7Adam/batch_normalization_38/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_38/beta/m/Read/ReadVariableOp+Adam/conv1d_39/kernel/m/Read/ReadVariableOp)Adam/conv1d_39/bias/m/Read/ReadVariableOp7Adam/batch_normalization_39/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_39/beta/m/Read/ReadVariableOp*Adam/dense_57/kernel/m/Read/ReadVariableOp(Adam/dense_57/bias/m/Read/ReadVariableOp*Adam/dense_58/kernel/m/Read/ReadVariableOp(Adam/dense_58/bias/m/Read/ReadVariableOp*Adam/dense_59/kernel/m/Read/ReadVariableOp(Adam/dense_59/bias/m/Read/ReadVariableOp+Adam/conv1d_38/kernel/v/Read/ReadVariableOp)Adam/conv1d_38/bias/v/Read/ReadVariableOp7Adam/batch_normalization_38/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_38/beta/v/Read/ReadVariableOp+Adam/conv1d_39/kernel/v/Read/ReadVariableOp)Adam/conv1d_39/bias/v/Read/ReadVariableOp7Adam/batch_normalization_39/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_39/beta/v/Read/ReadVariableOp*Adam/dense_57/kernel/v/Read/ReadVariableOp(Adam/dense_57/bias/v/Read/ReadVariableOp*Adam/dense_58/kernel/v/Read/ReadVariableOp(Adam/dense_58/bias/v/Read/ReadVariableOp*Adam/dense_59/kernel/v/Read/ReadVariableOp(Adam/dense_59/bias/v/Read/ReadVariableOpConst*D
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
__inference__traced_save_420852
Э
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_38/kernelconv1d_38/biasbatch_normalization_38/gammabatch_normalization_38/beta"batch_normalization_38/moving_mean&batch_normalization_38/moving_varianceconv1d_39/kernelconv1d_39/biasbatch_normalization_39/gammabatch_normalization_39/beta"batch_normalization_39/moving_mean&batch_normalization_39/moving_variancedense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1Adam/conv1d_38/kernel/mAdam/conv1d_38/bias/m#Adam/batch_normalization_38/gamma/m"Adam/batch_normalization_38/beta/mAdam/conv1d_39/kernel/mAdam/conv1d_39/bias/m#Adam/batch_normalization_39/gamma/m"Adam/batch_normalization_39/beta/mAdam/dense_57/kernel/mAdam/dense_57/bias/mAdam/dense_58/kernel/mAdam/dense_58/bias/mAdam/dense_59/kernel/mAdam/dense_59/bias/mAdam/conv1d_38/kernel/vAdam/conv1d_38/bias/v#Adam/batch_normalization_38/gamma/v"Adam/batch_normalization_38/beta/vAdam/conv1d_39/kernel/vAdam/conv1d_39/bias/v#Adam/batch_normalization_39/gamma/v"Adam/batch_normalization_39/beta/vAdam/dense_57/kernel/vAdam/dense_57/bias/vAdam/dense_58/kernel/vAdam/dense_58/bias/vAdam/dense_59/kernel/vAdam/dense_59/bias/v*C
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
"__inference__traced_restore_421027В
Ю
Њ
7__inference_batch_normalization_38_layer_call_fn_420350

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_4192402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџH::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs

M
1__inference_max_pooling1d_38_layer_call_fn_418994

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_38_layer_call_and_return_conditional_losses_4189882
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
щ

R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420324

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
:џџџџџџџџџH2
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
:џџџџџџџџџH2
batchnorm/add_1п
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџH::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
Щ
d
F__inference_dropout_39_layer_call_and_return_conditional_losses_420634

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_420587

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф	
н
D__inference_dense_58_layer_call_and_return_conditional_losses_420608

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
є0
Щ
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_419090

inputs
assignmovingavg_419065
assignmovingavg_1_419071)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
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
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
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
moments/Squeeze_1Ь
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/419065*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_419065*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/419065*
_output_shapes
:2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/419065*
_output_shapes
:2
AssignMovingAvg/mulЏ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_419065AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/419065*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/419071*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_419071*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/419071*
_output_shapes
:2
AssignMovingAvg_1/subђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/419071*
_output_shapes
:2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_419071AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/419071*
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
 :џџџџџџџџџџџџџџџџџџ2
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
 :џџџџџџџџџџџџџџџџџџ2
batchnorm/add_1Р
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


.__inference_sequential_19_layer_call_fn_419796
conv1d_38_input
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
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallconv1d_38_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_4197572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:џџџџџџџџџH
)
_user_specified_nameconv1d_38_input

e
F__inference_dropout_38_layer_call_and_return_conditional_losses_420582

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ј
E__inference_conv1d_38_layer_call_and_return_conditional_losses_419169

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџH2
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџH*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџH2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџH2
ReluЈ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџH2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџH::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
с
~
)__inference_dense_58_layer_call_fn_420617

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_4194832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф	
н
D__inference_dense_58_layer_call_and_return_conditional_losses_419483

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ј
E__inference_conv1d_39_layer_call_and_return_conditional_losses_419293

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$2
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ$2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$2
ReluЈ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ$
 
_user_specified_nameinputs


R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420431

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 :џџџџџџџџџџџџџџџџџџ2
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
 :џџџџџџџџџџџџџџџџџџ2
batchnorm/add_1ш
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
є0
Щ
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_418935

inputs
assignmovingavg_418910
assignmovingavg_1_418916)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
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
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
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
moments/Squeeze_1Ь
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/418910*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_418910*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/418910*
_output_shapes
:2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/418910*
_output_shapes
:2
AssignMovingAvg/mulЏ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_418910AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/418910*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/418916*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_418916*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/418916*
_output_shapes
:2
AssignMovingAvg_1/subђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/418916*
_output_shapes
:2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_418916AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/418916*
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
 :џџџџџџџџџџџџџџџџџџ2
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
 :џџџџџџџџџџџџџџџџџџ2
batchnorm/add_1Р
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
є0
Щ
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420222

inputs
assignmovingavg_420197
assignmovingavg_1_420203)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
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
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
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
moments/Squeeze_1Ь
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/420197*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_420197*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/420197*
_output_shapes
:2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/420197*
_output_shapes
:2
AssignMovingAvg/mulЏ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_420197AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/420197*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/420203*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_420203*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/420203*
_output_shapes
:2
AssignMovingAvg_1/subђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/420203*
_output_shapes
:2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_420203AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/420203*
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
 :џџџџџџџџџџџџџџџџџџ2
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
 :џџџџџџџџџџџџџџџџџџ2
batchnorm/add_1Р
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н0
Щ
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420493

inputs
assignmovingavg_420468
assignmovingavg_1_420474)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
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
moments/StopGradientЈ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
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
moments/Squeeze_1Ь
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/420468*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_420468*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/420468*
_output_shapes
:2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/420468*
_output_shapes
:2
AssignMovingAvg/mulЏ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_420468AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/420468*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/420474*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_420474*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/420474*
_output_shapes
:2
AssignMovingAvg_1/subђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/420474*
_output_shapes
:2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_420474AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/420474*
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
:џџџџџџџџџ$2
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
:џџџџџџџџџ$2
batchnorm/add_1З
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ$::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ$
 
_user_specified_nameinputs
Ў
Г
!__inference__wrapped_model_418839
conv1d_38_inputG
Csequential_19_conv1d_38_conv1d_expanddims_1_readvariableop_resource;
7sequential_19_conv1d_38_biasadd_readvariableop_resourceJ
Fsequential_19_batch_normalization_38_batchnorm_readvariableop_resourceN
Jsequential_19_batch_normalization_38_batchnorm_mul_readvariableop_resourceL
Hsequential_19_batch_normalization_38_batchnorm_readvariableop_1_resourceL
Hsequential_19_batch_normalization_38_batchnorm_readvariableop_2_resourceG
Csequential_19_conv1d_39_conv1d_expanddims_1_readvariableop_resource;
7sequential_19_conv1d_39_biasadd_readvariableop_resourceJ
Fsequential_19_batch_normalization_39_batchnorm_readvariableop_resourceN
Jsequential_19_batch_normalization_39_batchnorm_mul_readvariableop_resourceL
Hsequential_19_batch_normalization_39_batchnorm_readvariableop_1_resourceL
Hsequential_19_batch_normalization_39_batchnorm_readvariableop_2_resource9
5sequential_19_dense_57_matmul_readvariableop_resource:
6sequential_19_dense_57_biasadd_readvariableop_resource9
5sequential_19_dense_58_matmul_readvariableop_resource:
6sequential_19_dense_58_biasadd_readvariableop_resource9
5sequential_19_dense_59_matmul_readvariableop_resource:
6sequential_19_dense_59_biasadd_readvariableop_resource
identityЂ=sequential_19/batch_normalization_38/batchnorm/ReadVariableOpЂ?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_1Ђ?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_2ЂAsequential_19/batch_normalization_38/batchnorm/mul/ReadVariableOpЂ=sequential_19/batch_normalization_39/batchnorm/ReadVariableOpЂ?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_1Ђ?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_2ЂAsequential_19/batch_normalization_39/batchnorm/mul/ReadVariableOpЂ.sequential_19/conv1d_38/BiasAdd/ReadVariableOpЂ:sequential_19/conv1d_38/conv1d/ExpandDims_1/ReadVariableOpЂ.sequential_19/conv1d_39/BiasAdd/ReadVariableOpЂ:sequential_19/conv1d_39/conv1d/ExpandDims_1/ReadVariableOpЂ-sequential_19/dense_57/BiasAdd/ReadVariableOpЂ,sequential_19/dense_57/MatMul/ReadVariableOpЂ-sequential_19/dense_58/BiasAdd/ReadVariableOpЂ,sequential_19/dense_58/MatMul/ReadVariableOpЂ-sequential_19/dense_59/BiasAdd/ReadVariableOpЂ,sequential_19/dense_59/MatMul/ReadVariableOpЉ
-sequential_19/conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-sequential_19/conv1d_38/conv1d/ExpandDims/dimч
)sequential_19/conv1d_38/conv1d/ExpandDims
ExpandDimsconv1d_38_input6sequential_19/conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџH2+
)sequential_19/conv1d_38/conv1d/ExpandDims
:sequential_19/conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_19_conv1d_38_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02<
:sequential_19/conv1d_38/conv1d/ExpandDims_1/ReadVariableOpЄ
/sequential_19/conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_19/conv1d_38/conv1d/ExpandDims_1/dim
+sequential_19/conv1d_38/conv1d/ExpandDims_1
ExpandDimsBsequential_19/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_19/conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2-
+sequential_19/conv1d_38/conv1d/ExpandDims_1
sequential_19/conv1d_38/conv1dConv2D2sequential_19/conv1d_38/conv1d/ExpandDims:output:04sequential_19/conv1d_38/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
paddingSAME*
strides
2 
sequential_19/conv1d_38/conv1dк
&sequential_19/conv1d_38/conv1d/SqueezeSqueeze'sequential_19/conv1d_38/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџH*
squeeze_dims

§џџџџџџџџ2(
&sequential_19/conv1d_38/conv1d/Squeezeд
.sequential_19/conv1d_38/BiasAdd/ReadVariableOpReadVariableOp7sequential_19_conv1d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_19/conv1d_38/BiasAdd/ReadVariableOpь
sequential_19/conv1d_38/BiasAddBiasAdd/sequential_19/conv1d_38/conv1d/Squeeze:output:06sequential_19/conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџH2!
sequential_19/conv1d_38/BiasAddЄ
sequential_19/conv1d_38/ReluRelu(sequential_19/conv1d_38/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџH2
sequential_19/conv1d_38/Relu
=sequential_19/batch_normalization_38/batchnorm/ReadVariableOpReadVariableOpFsequential_19_batch_normalization_38_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_19/batch_normalization_38/batchnorm/ReadVariableOpБ
4sequential_19/batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4sequential_19/batch_normalization_38/batchnorm/add/y
2sequential_19/batch_normalization_38/batchnorm/addAddV2Esequential_19/batch_normalization_38/batchnorm/ReadVariableOp:value:0=sequential_19/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_19/batch_normalization_38/batchnorm/addв
4sequential_19/batch_normalization_38/batchnorm/RsqrtRsqrt6sequential_19/batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_19/batch_normalization_38/batchnorm/Rsqrt
Asequential_19/batch_normalization_38/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_19_batch_normalization_38_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_19/batch_normalization_38/batchnorm/mul/ReadVariableOp
2sequential_19/batch_normalization_38/batchnorm/mulMul8sequential_19/batch_normalization_38/batchnorm/Rsqrt:y:0Isequential_19/batch_normalization_38/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_19/batch_normalization_38/batchnorm/mul
4sequential_19/batch_normalization_38/batchnorm/mul_1Mul*sequential_19/conv1d_38/Relu:activations:06sequential_19/batch_normalization_38/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџH26
4sequential_19/batch_normalization_38/batchnorm/mul_1
?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_19_batch_normalization_38_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_1
4sequential_19/batch_normalization_38/batchnorm/mul_2MulGsequential_19/batch_normalization_38/batchnorm/ReadVariableOp_1:value:06sequential_19/batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_19/batch_normalization_38/batchnorm/mul_2
?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_19_batch_normalization_38_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_2
2sequential_19/batch_normalization_38/batchnorm/subSubGsequential_19/batch_normalization_38/batchnorm/ReadVariableOp_2:value:08sequential_19/batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_19/batch_normalization_38/batchnorm/sub
4sequential_19/batch_normalization_38/batchnorm/add_1AddV28sequential_19/batch_normalization_38/batchnorm/mul_1:z:06sequential_19/batch_normalization_38/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџH26
4sequential_19/batch_normalization_38/batchnorm/add_1 
-sequential_19/max_pooling1d_38/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_19/max_pooling1d_38/ExpandDims/dim
)sequential_19/max_pooling1d_38/ExpandDims
ExpandDims8sequential_19/batch_normalization_38/batchnorm/add_1:z:06sequential_19/max_pooling1d_38/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџH2+
)sequential_19/max_pooling1d_38/ExpandDimsћ
&sequential_19/max_pooling1d_38/MaxPoolMaxPool2sequential_19/max_pooling1d_38/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ$*
ksize
*
paddingSAME*
strides
2(
&sequential_19/max_pooling1d_38/MaxPoolй
&sequential_19/max_pooling1d_38/SqueezeSqueeze/sequential_19/max_pooling1d_38/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$*
squeeze_dims
2(
&sequential_19/max_pooling1d_38/SqueezeЉ
-sequential_19/conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2/
-sequential_19/conv1d_39/conv1d/ExpandDims/dim
)sequential_19/conv1d_39/conv1d/ExpandDims
ExpandDims/sequential_19/max_pooling1d_38/Squeeze:output:06sequential_19/conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$2+
)sequential_19/conv1d_39/conv1d/ExpandDims
:sequential_19/conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_19_conv1d_39_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02<
:sequential_19/conv1d_39/conv1d/ExpandDims_1/ReadVariableOpЄ
/sequential_19/conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_19/conv1d_39/conv1d/ExpandDims_1/dim
+sequential_19/conv1d_39/conv1d/ExpandDims_1
ExpandDimsBsequential_19/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_19/conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2-
+sequential_19/conv1d_39/conv1d/ExpandDims_1
sequential_19/conv1d_39/conv1dConv2D2sequential_19/conv1d_39/conv1d/ExpandDims:output:04sequential_19/conv1d_39/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$*
paddingSAME*
strides
2 
sequential_19/conv1d_39/conv1dк
&sequential_19/conv1d_39/conv1d/SqueezeSqueeze'sequential_19/conv1d_39/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$*
squeeze_dims

§џџџџџџџџ2(
&sequential_19/conv1d_39/conv1d/Squeezeд
.sequential_19/conv1d_39/BiasAdd/ReadVariableOpReadVariableOp7sequential_19_conv1d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_19/conv1d_39/BiasAdd/ReadVariableOpь
sequential_19/conv1d_39/BiasAddBiasAdd/sequential_19/conv1d_39/conv1d/Squeeze:output:06sequential_19/conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ$2!
sequential_19/conv1d_39/BiasAddЄ
sequential_19/conv1d_39/ReluRelu(sequential_19/conv1d_39/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$2
sequential_19/conv1d_39/Relu
=sequential_19/batch_normalization_39/batchnorm/ReadVariableOpReadVariableOpFsequential_19_batch_normalization_39_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential_19/batch_normalization_39/batchnorm/ReadVariableOpБ
4sequential_19/batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4sequential_19/batch_normalization_39/batchnorm/add/y
2sequential_19/batch_normalization_39/batchnorm/addAddV2Esequential_19/batch_normalization_39/batchnorm/ReadVariableOp:value:0=sequential_19/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2sequential_19/batch_normalization_39/batchnorm/addв
4sequential_19/batch_normalization_39/batchnorm/RsqrtRsqrt6sequential_19/batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes
:26
4sequential_19/batch_normalization_39/batchnorm/Rsqrt
Asequential_19/batch_normalization_39/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_19_batch_normalization_39_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_19/batch_normalization_39/batchnorm/mul/ReadVariableOp
2sequential_19/batch_normalization_39/batchnorm/mulMul8sequential_19/batch_normalization_39/batchnorm/Rsqrt:y:0Isequential_19/batch_normalization_39/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2sequential_19/batch_normalization_39/batchnorm/mul
4sequential_19/batch_normalization_39/batchnorm/mul_1Mul*sequential_19/conv1d_39/Relu:activations:06sequential_19/batch_normalization_39/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ$26
4sequential_19/batch_normalization_39/batchnorm/mul_1
?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_19_batch_normalization_39_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_1
4sequential_19/batch_normalization_39/batchnorm/mul_2MulGsequential_19/batch_normalization_39/batchnorm/ReadVariableOp_1:value:06sequential_19/batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4sequential_19/batch_normalization_39/batchnorm/mul_2
?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_19_batch_normalization_39_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_2
2sequential_19/batch_normalization_39/batchnorm/subSubGsequential_19/batch_normalization_39/batchnorm/ReadVariableOp_2:value:08sequential_19/batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2sequential_19/batch_normalization_39/batchnorm/sub
4sequential_19/batch_normalization_39/batchnorm/add_1AddV28sequential_19/batch_normalization_39/batchnorm/mul_1:z:06sequential_19/batch_normalization_39/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ$26
4sequential_19/batch_normalization_39/batchnorm/add_1 
-sequential_19/max_pooling1d_39/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_19/max_pooling1d_39/ExpandDims/dim
)sequential_19/max_pooling1d_39/ExpandDims
ExpandDims8sequential_19/batch_normalization_39/batchnorm/add_1:z:06sequential_19/max_pooling1d_39/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$2+
)sequential_19/max_pooling1d_39/ExpandDimsћ
&sequential_19/max_pooling1d_39/MaxPoolMaxPool2sequential_19/max_pooling1d_39/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2(
&sequential_19/max_pooling1d_39/MaxPoolй
&sequential_19/max_pooling1d_39/SqueezeSqueeze/sequential_19/max_pooling1d_39/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2(
&sequential_19/max_pooling1d_39/Squeeze
sequential_19/flatten_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџи   2 
sequential_19/flatten_19/Constм
 sequential_19/flatten_19/ReshapeReshape/sequential_19/max_pooling1d_39/Squeeze:output:0'sequential_19/flatten_19/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџи2"
 sequential_19/flatten_19/Reshapeг
,sequential_19/dense_57/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_57_matmul_readvariableop_resource*
_output_shapes
:	и*
dtype02.
,sequential_19/dense_57/MatMul/ReadVariableOpл
sequential_19/dense_57/MatMulMatMul)sequential_19/flatten_19/Reshape:output:04sequential_19/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_19/dense_57/MatMulб
-sequential_19/dense_57/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_19/dense_57/BiasAdd/ReadVariableOpн
sequential_19/dense_57/BiasAddBiasAdd'sequential_19/dense_57/MatMul:product:05sequential_19/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_19/dense_57/BiasAdd
sequential_19/dense_57/TanhTanh'sequential_19/dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_19/dense_57/TanhЅ
!sequential_19/dropout_38/IdentityIdentitysequential_19/dense_57/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!sequential_19/dropout_38/Identityв
,sequential_19/dense_58/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_58_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_19/dense_58/MatMul/ReadVariableOpм
sequential_19/dense_58/MatMulMatMul*sequential_19/dropout_38/Identity:output:04sequential_19/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_19/dense_58/MatMulб
-sequential_19/dense_58/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_19/dense_58/BiasAdd/ReadVariableOpн
sequential_19/dense_58/BiasAddBiasAdd'sequential_19/dense_58/MatMul:product:05sequential_19/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_19/dense_58/BiasAdd
sequential_19/dense_58/TanhTanh'sequential_19/dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_19/dense_58/TanhЅ
!sequential_19/dropout_39/IdentityIdentitysequential_19/dense_58/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!sequential_19/dropout_39/Identityв
,sequential_19/dense_59/MatMul/ReadVariableOpReadVariableOp5sequential_19_dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,sequential_19/dense_59/MatMul/ReadVariableOpм
sequential_19/dense_59/MatMulMatMul*sequential_19/dropout_39/Identity:output:04sequential_19/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_19/dense_59/MatMulб
-sequential_19/dense_59/BiasAdd/ReadVariableOpReadVariableOp6sequential_19_dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_19/dense_59/BiasAdd/ReadVariableOpн
sequential_19/dense_59/BiasAddBiasAdd'sequential_19/dense_59/MatMul:product:05sequential_19/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_19/dense_59/BiasAddІ
sequential_19/dense_59/SoftmaxSoftmax'sequential_19/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential_19/dense_59/Softmax	
IdentityIdentity(sequential_19/dense_59/Softmax:softmax:0>^sequential_19/batch_normalization_38/batchnorm/ReadVariableOp@^sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_1@^sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_2B^sequential_19/batch_normalization_38/batchnorm/mul/ReadVariableOp>^sequential_19/batch_normalization_39/batchnorm/ReadVariableOp@^sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_1@^sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_2B^sequential_19/batch_normalization_39/batchnorm/mul/ReadVariableOp/^sequential_19/conv1d_38/BiasAdd/ReadVariableOp;^sequential_19/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp/^sequential_19/conv1d_39/BiasAdd/ReadVariableOp;^sequential_19/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp.^sequential_19/dense_57/BiasAdd/ReadVariableOp-^sequential_19/dense_57/MatMul/ReadVariableOp.^sequential_19/dense_58/BiasAdd/ReadVariableOp-^sequential_19/dense_58/MatMul/ReadVariableOp.^sequential_19/dense_59/BiasAdd/ReadVariableOp-^sequential_19/dense_59/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::2~
=sequential_19/batch_normalization_38/batchnorm/ReadVariableOp=sequential_19/batch_normalization_38/batchnorm/ReadVariableOp2
?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_1?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_12
?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_2?sequential_19/batch_normalization_38/batchnorm/ReadVariableOp_22
Asequential_19/batch_normalization_38/batchnorm/mul/ReadVariableOpAsequential_19/batch_normalization_38/batchnorm/mul/ReadVariableOp2~
=sequential_19/batch_normalization_39/batchnorm/ReadVariableOp=sequential_19/batch_normalization_39/batchnorm/ReadVariableOp2
?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_1?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_12
?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_2?sequential_19/batch_normalization_39/batchnorm/ReadVariableOp_22
Asequential_19/batch_normalization_39/batchnorm/mul/ReadVariableOpAsequential_19/batch_normalization_39/batchnorm/mul/ReadVariableOp2`
.sequential_19/conv1d_38/BiasAdd/ReadVariableOp.sequential_19/conv1d_38/BiasAdd/ReadVariableOp2x
:sequential_19/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:sequential_19/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp2`
.sequential_19/conv1d_39/BiasAdd/ReadVariableOp.sequential_19/conv1d_39/BiasAdd/ReadVariableOp2x
:sequential_19/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:sequential_19/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp2^
-sequential_19/dense_57/BiasAdd/ReadVariableOp-sequential_19/dense_57/BiasAdd/ReadVariableOp2\
,sequential_19/dense_57/MatMul/ReadVariableOp,sequential_19/dense_57/MatMul/ReadVariableOp2^
-sequential_19/dense_58/BiasAdd/ReadVariableOp-sequential_19/dense_58/BiasAdd/ReadVariableOp2\
,sequential_19/dense_58/MatMul/ReadVariableOp,sequential_19/dense_58/MatMul/ReadVariableOp2^
-sequential_19/dense_59/BiasAdd/ReadVariableOp-sequential_19/dense_59/BiasAdd/ReadVariableOp2\
,sequential_19/dense_59/MatMul/ReadVariableOp,sequential_19/dense_59/MatMul/ReadVariableOp:\ X
+
_output_shapes
:џџџџџџџџџH
)
_user_specified_nameconv1d_38_input
ы

љ
$__inference_signature_wrapper_419847
conv1d_38_input
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
identityЂStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallconv1d_38_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_4188392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:џџџџџџџџџH
)
_user_specified_nameconv1d_38_input
У=
М
I__inference_sequential_19_layer_call_and_return_conditional_losses_419557
conv1d_38_input
conv1d_38_419180
conv1d_38_419182!
batch_normalization_38_419267!
batch_normalization_38_419269!
batch_normalization_38_419271!
batch_normalization_38_419273
conv1d_39_419304
conv1d_39_419306!
batch_normalization_39_419391!
batch_normalization_39_419393!
batch_normalization_39_419395!
batch_normalization_39_419397
dense_57_419437
dense_57_419439
dense_58_419494
dense_58_419496
dense_59_419551
dense_59_419553
identityЂ.batch_normalization_38/StatefulPartitionedCallЂ.batch_normalization_39/StatefulPartitionedCallЂ!conv1d_38/StatefulPartitionedCallЂ!conv1d_39/StatefulPartitionedCallЂ dense_57/StatefulPartitionedCallЂ dense_58/StatefulPartitionedCallЂ dense_59/StatefulPartitionedCallЂ"dropout_38/StatefulPartitionedCallЂ"dropout_39/StatefulPartitionedCallЋ
!conv1d_38/StatefulPartitionedCallStatefulPartitionedCallconv1d_38_inputconv1d_38_419180conv1d_38_419182*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_38_layer_call_and_return_conditional_losses_4191692#
!conv1d_38/StatefulPartitionedCallЧ
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall*conv1d_38/StatefulPartitionedCall:output:0batch_normalization_38_419267batch_normalization_38_419269batch_normalization_38_419271batch_normalization_38_419273*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_41922020
.batch_normalization_38/StatefulPartitionedCallІ
 max_pooling1d_38/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_38_layer_call_and_return_conditional_losses_4189882"
 max_pooling1d_38/PartitionedCallХ
!conv1d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_38/PartitionedCall:output:0conv1d_39_419304conv1d_39_419306*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_39_layer_call_and_return_conditional_losses_4192932#
!conv1d_39/StatefulPartitionedCallЧ
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv1d_39/StatefulPartitionedCall:output:0batch_normalization_39_419391batch_normalization_39_419393batch_normalization_39_419395batch_normalization_39_419397*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_41934420
.batch_normalization_39/StatefulPartitionedCallІ
 max_pooling1d_39/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_39_layer_call_and_return_conditional_losses_4191432"
 max_pooling1d_39/PartitionedCall
flatten_19/PartitionedCallPartitionedCall)max_pooling1d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџи* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_4194072
flatten_19/PartitionedCallЖ
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#flatten_19/PartitionedCall:output:0dense_57_419437dense_57_419439*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_4194262"
 dense_57/StatefulPartitionedCall
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_4194542$
"dropout_38/StatefulPartitionedCallО
 dense_58/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_58_419494dense_58_419496*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_4194832"
 dense_58/StatefulPartitionedCallП
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0#^dropout_38/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_4195112$
"dropout_39/StatefulPartitionedCallО
 dense_59/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0dense_59_419551dense_59_419553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_4195402"
 dense_59/StatefulPartitionedCallк
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall"^conv1d_38/StatefulPartitionedCall"^conv1d_39/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2F
!conv1d_38/StatefulPartitionedCall!conv1d_38/StatefulPartitionedCall2F
!conv1d_39/StatefulPartitionedCall!conv1d_39/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall:\ X
+
_output_shapes
:џџџџџџџџџH
)
_user_specified_nameconv1d_38_input
щ

R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_419364

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
:џџџџџџџџџ$2
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
:џџџџџџџџџ$2
batchnorm/add_1п
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ$::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ$
 
_user_specified_nameinputs
Ь
Њ
7__inference_batch_normalization_38_layer_call_fn_420337

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_4192202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџH::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
:
щ
I__inference_sequential_19_layer_call_and_return_conditional_losses_419757

inputs
conv1d_38_419708
conv1d_38_419710!
batch_normalization_38_419713!
batch_normalization_38_419715!
batch_normalization_38_419717!
batch_normalization_38_419719
conv1d_39_419723
conv1d_39_419725!
batch_normalization_39_419728!
batch_normalization_39_419730!
batch_normalization_39_419732!
batch_normalization_39_419734
dense_57_419739
dense_57_419741
dense_58_419745
dense_58_419747
dense_59_419751
dense_59_419753
identityЂ.batch_normalization_38/StatefulPartitionedCallЂ.batch_normalization_39/StatefulPartitionedCallЂ!conv1d_38/StatefulPartitionedCallЂ!conv1d_39/StatefulPartitionedCallЂ dense_57/StatefulPartitionedCallЂ dense_58/StatefulPartitionedCallЂ dense_59/StatefulPartitionedCallЂ
!conv1d_38/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_38_419708conv1d_38_419710*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_38_layer_call_and_return_conditional_losses_4191692#
!conv1d_38/StatefulPartitionedCallЩ
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall*conv1d_38/StatefulPartitionedCall:output:0batch_normalization_38_419713batch_normalization_38_419715batch_normalization_38_419717batch_normalization_38_419719*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_41924020
.batch_normalization_38/StatefulPartitionedCallІ
 max_pooling1d_38/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_38_layer_call_and_return_conditional_losses_4189882"
 max_pooling1d_38/PartitionedCallХ
!conv1d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_38/PartitionedCall:output:0conv1d_39_419723conv1d_39_419725*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_39_layer_call_and_return_conditional_losses_4192932#
!conv1d_39/StatefulPartitionedCallЩ
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv1d_39/StatefulPartitionedCall:output:0batch_normalization_39_419728batch_normalization_39_419730batch_normalization_39_419732batch_normalization_39_419734*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_41936420
.batch_normalization_39/StatefulPartitionedCallІ
 max_pooling1d_39/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_39_layer_call_and_return_conditional_losses_4191432"
 max_pooling1d_39/PartitionedCall
flatten_19/PartitionedCallPartitionedCall)max_pooling1d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџи* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_4194072
flatten_19/PartitionedCallЖ
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#flatten_19/PartitionedCall:output:0dense_57_419739dense_57_419741*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_4194262"
 dense_57/StatefulPartitionedCall
dropout_38/PartitionedCallPartitionedCall)dense_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_4194592
dropout_38/PartitionedCallЖ
 dense_58/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_58_419745dense_58_419747*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_4194832"
 dense_58/StatefulPartitionedCall
dropout_39/PartitionedCallPartitionedCall)dense_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_4195162
dropout_39/PartitionedCallЖ
 dense_59/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0dense_59_419751dense_59_419753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_4195402"
 dense_59/StatefulPartitionedCall
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall"^conv1d_38/StatefulPartitionedCall"^conv1d_39/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2F
!conv1d_38/StatefulPartitionedCall!conv1d_38/StatefulPartitionedCall2F
!conv1d_39/StatefulPartitionedCall!conv1d_39/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
Н0
Щ
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_419220

inputs
assignmovingavg_419195
assignmovingavg_1_419201)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
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
moments/StopGradientЈ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџH2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
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
moments/Squeeze_1Ь
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/419195*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_419195*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/419195*
_output_shapes
:2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/419195*
_output_shapes
:2
AssignMovingAvg/mulЏ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_419195AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/419195*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/419201*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_419201*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/419201*
_output_shapes
:2
AssignMovingAvg_1/subђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/419201*
_output_shapes
:2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_419201AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/419201*
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
:џџџџџџџџџH2
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
:џџџџџџџџџH2
batchnorm/add_1З
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџH::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
о
к
I__inference_sequential_19_layer_call_and_return_conditional_losses_420079

inputs9
5conv1d_38_conv1d_expanddims_1_readvariableop_resource-
)conv1d_38_biasadd_readvariableop_resource<
8batch_normalization_38_batchnorm_readvariableop_resource@
<batch_normalization_38_batchnorm_mul_readvariableop_resource>
:batch_normalization_38_batchnorm_readvariableop_1_resource>
:batch_normalization_38_batchnorm_readvariableop_2_resource9
5conv1d_39_conv1d_expanddims_1_readvariableop_resource-
)conv1d_39_biasadd_readvariableop_resource<
8batch_normalization_39_batchnorm_readvariableop_resource@
<batch_normalization_39_batchnorm_mul_readvariableop_resource>
:batch_normalization_39_batchnorm_readvariableop_1_resource>
:batch_normalization_39_batchnorm_readvariableop_2_resource+
'dense_57_matmul_readvariableop_resource,
(dense_57_biasadd_readvariableop_resource+
'dense_58_matmul_readvariableop_resource,
(dense_58_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource
identityЂ/batch_normalization_38/batchnorm/ReadVariableOpЂ1batch_normalization_38/batchnorm/ReadVariableOp_1Ђ1batch_normalization_38/batchnorm/ReadVariableOp_2Ђ3batch_normalization_38/batchnorm/mul/ReadVariableOpЂ/batch_normalization_39/batchnorm/ReadVariableOpЂ1batch_normalization_39/batchnorm/ReadVariableOp_1Ђ1batch_normalization_39/batchnorm/ReadVariableOp_2Ђ3batch_normalization_39/batchnorm/mul/ReadVariableOpЂ conv1d_38/BiasAdd/ReadVariableOpЂ,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpЂ conv1d_39/BiasAdd/ReadVariableOpЂ,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpЂdense_57/BiasAdd/ReadVariableOpЂdense_57/MatMul/ReadVariableOpЂdense_58/BiasAdd/ReadVariableOpЂdense_58/MatMul/ReadVariableOpЂdense_59/BiasAdd/ReadVariableOpЂdense_59/MatMul/ReadVariableOp
conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2!
conv1d_38/conv1d/ExpandDims/dimД
conv1d_38/conv1d/ExpandDims
ExpandDimsinputs(conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџH2
conv1d_38/conv1d/ExpandDimsж
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_38/conv1d/ExpandDims_1/dimп
conv1d_38/conv1d/ExpandDims_1
ExpandDims4conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_38/conv1d/ExpandDims_1о
conv1d_38/conv1dConv2D$conv1d_38/conv1d/ExpandDims:output:0&conv1d_38/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
paddingSAME*
strides
2
conv1d_38/conv1dА
conv1d_38/conv1d/SqueezeSqueezeconv1d_38/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџH*
squeeze_dims

§џџџџџџџџ2
conv1d_38/conv1d/SqueezeЊ
 conv1d_38/BiasAdd/ReadVariableOpReadVariableOp)conv1d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_38/BiasAdd/ReadVariableOpД
conv1d_38/BiasAddBiasAdd!conv1d_38/conv1d/Squeeze:output:0(conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџH2
conv1d_38/BiasAddz
conv1d_38/ReluReluconv1d_38/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџH2
conv1d_38/Reluз
/batch_normalization_38/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_38_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_38/batchnorm/ReadVariableOp
&batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_38/batchnorm/add/yф
$batch_normalization_38/batchnorm/addAddV27batch_normalization_38/batchnorm/ReadVariableOp:value:0/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/addЈ
&batch_normalization_38/batchnorm/RsqrtRsqrt(batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/Rsqrtу
3batch_normalization_38/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_38_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_38/batchnorm/mul/ReadVariableOpс
$batch_normalization_38/batchnorm/mulMul*batch_normalization_38/batchnorm/Rsqrt:y:0;batch_normalization_38/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/mulе
&batch_normalization_38/batchnorm/mul_1Mulconv1d_38/Relu:activations:0(batch_normalization_38/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџH2(
&batch_normalization_38/batchnorm/mul_1н
1batch_normalization_38/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_38_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_38/batchnorm/ReadVariableOp_1с
&batch_normalization_38/batchnorm/mul_2Mul9batch_normalization_38/batchnorm/ReadVariableOp_1:value:0(batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/mul_2н
1batch_normalization_38/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_38_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_38/batchnorm/ReadVariableOp_2п
$batch_normalization_38/batchnorm/subSub9batch_normalization_38/batchnorm/ReadVariableOp_2:value:0*batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/subх
&batch_normalization_38/batchnorm/add_1AddV2*batch_normalization_38/batchnorm/mul_1:z:0(batch_normalization_38/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџH2(
&batch_normalization_38/batchnorm/add_1
max_pooling1d_38/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_38/ExpandDims/dimи
max_pooling1d_38/ExpandDims
ExpandDims*batch_normalization_38/batchnorm/add_1:z:0(max_pooling1d_38/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџH2
max_pooling1d_38/ExpandDimsб
max_pooling1d_38/MaxPoolMaxPool$max_pooling1d_38/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ$*
ksize
*
paddingSAME*
strides
2
max_pooling1d_38/MaxPoolЏ
max_pooling1d_38/SqueezeSqueeze!max_pooling1d_38/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$*
squeeze_dims
2
max_pooling1d_38/Squeeze
conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2!
conv1d_39/conv1d/ExpandDims/dimЯ
conv1d_39/conv1d/ExpandDims
ExpandDims!max_pooling1d_38/Squeeze:output:0(conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$2
conv1d_39/conv1d/ExpandDimsж
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_39_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_39/conv1d/ExpandDims_1/dimп
conv1d_39/conv1d/ExpandDims_1
ExpandDims4conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_39/conv1d/ExpandDims_1о
conv1d_39/conv1dConv2D$conv1d_39/conv1d/ExpandDims:output:0&conv1d_39/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$*
paddingSAME*
strides
2
conv1d_39/conv1dА
conv1d_39/conv1d/SqueezeSqueezeconv1d_39/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$*
squeeze_dims

§џџџџџџџџ2
conv1d_39/conv1d/SqueezeЊ
 conv1d_39/BiasAdd/ReadVariableOpReadVariableOp)conv1d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_39/BiasAdd/ReadVariableOpД
conv1d_39/BiasAddBiasAdd!conv1d_39/conv1d/Squeeze:output:0(conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ$2
conv1d_39/BiasAddz
conv1d_39/ReluReluconv1d_39/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$2
conv1d_39/Reluз
/batch_normalization_39/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_39_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_39/batchnorm/ReadVariableOp
&batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_39/batchnorm/add/yф
$batch_normalization_39/batchnorm/addAddV27batch_normalization_39/batchnorm/ReadVariableOp:value:0/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/addЈ
&batch_normalization_39/batchnorm/RsqrtRsqrt(batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/Rsqrtу
3batch_normalization_39/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_39_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_39/batchnorm/mul/ReadVariableOpс
$batch_normalization_39/batchnorm/mulMul*batch_normalization_39/batchnorm/Rsqrt:y:0;batch_normalization_39/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/mulе
&batch_normalization_39/batchnorm/mul_1Mulconv1d_39/Relu:activations:0(batch_normalization_39/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ$2(
&batch_normalization_39/batchnorm/mul_1н
1batch_normalization_39/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_39_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_39/batchnorm/ReadVariableOp_1с
&batch_normalization_39/batchnorm/mul_2Mul9batch_normalization_39/batchnorm/ReadVariableOp_1:value:0(batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/mul_2н
1batch_normalization_39/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_39_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_39/batchnorm/ReadVariableOp_2п
$batch_normalization_39/batchnorm/subSub9batch_normalization_39/batchnorm/ReadVariableOp_2:value:0*batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/subх
&batch_normalization_39/batchnorm/add_1AddV2*batch_normalization_39/batchnorm/mul_1:z:0(batch_normalization_39/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ$2(
&batch_normalization_39/batchnorm/add_1
max_pooling1d_39/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_39/ExpandDims/dimи
max_pooling1d_39/ExpandDims
ExpandDims*batch_normalization_39/batchnorm/add_1:z:0(max_pooling1d_39/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$2
max_pooling1d_39/ExpandDimsб
max_pooling1d_39/MaxPoolMaxPool$max_pooling1d_39/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
max_pooling1d_39/MaxPoolЏ
max_pooling1d_39/SqueezeSqueeze!max_pooling1d_39/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
max_pooling1d_39/Squeezeu
flatten_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџи   2
flatten_19/ConstЄ
flatten_19/ReshapeReshape!max_pooling1d_39/Squeeze:output:0flatten_19/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџи2
flatten_19/ReshapeЉ
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes
:	и*
dtype02 
dense_57/MatMul/ReadVariableOpЃ
dense_57/MatMulMatMulflatten_19/Reshape:output:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_57/MatMulЇ
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_57/BiasAdd/ReadVariableOpЅ
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_57/BiasAdds
dense_57/TanhTanhdense_57/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_57/Tanh{
dropout_38/IdentityIdentitydense_57/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_38/IdentityЈ
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_58/MatMul/ReadVariableOpЄ
dense_58/MatMulMatMuldropout_38/Identity:output:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_58/MatMulЇ
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_58/BiasAdd/ReadVariableOpЅ
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_58/BiasAdds
dense_58/TanhTanhdense_58/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_58/Tanh{
dropout_39/IdentityIdentitydense_58/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_39/IdentityЈ
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_59/MatMul/ReadVariableOpЄ
dense_59/MatMulMatMuldropout_39/Identity:output:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_59/MatMulЇ
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_59/BiasAdd/ReadVariableOpЅ
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_59/BiasAdd|
dense_59/SoftmaxSoftmaxdense_59/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_59/Softmaxћ
IdentityIdentitydense_59/Softmax:softmax:00^batch_normalization_38/batchnorm/ReadVariableOp2^batch_normalization_38/batchnorm/ReadVariableOp_12^batch_normalization_38/batchnorm/ReadVariableOp_24^batch_normalization_38/batchnorm/mul/ReadVariableOp0^batch_normalization_39/batchnorm/ReadVariableOp2^batch_normalization_39/batchnorm/ReadVariableOp_12^batch_normalization_39/batchnorm/ReadVariableOp_24^batch_normalization_39/batchnorm/mul/ReadVariableOp!^conv1d_38/BiasAdd/ReadVariableOp-^conv1d_38/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_39/BiasAdd/ReadVariableOp-^conv1d_39/conv1d/ExpandDims_1/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::2b
/batch_normalization_38/batchnorm/ReadVariableOp/batch_normalization_38/batchnorm/ReadVariableOp2f
1batch_normalization_38/batchnorm/ReadVariableOp_11batch_normalization_38/batchnorm/ReadVariableOp_12f
1batch_normalization_38/batchnorm/ReadVariableOp_21batch_normalization_38/batchnorm/ReadVariableOp_22j
3batch_normalization_38/batchnorm/mul/ReadVariableOp3batch_normalization_38/batchnorm/mul/ReadVariableOp2b
/batch_normalization_39/batchnorm/ReadVariableOp/batch_normalization_39/batchnorm/ReadVariableOp2f
1batch_normalization_39/batchnorm/ReadVariableOp_11batch_normalization_39/batchnorm/ReadVariableOp_12f
1batch_normalization_39/batchnorm/ReadVariableOp_21batch_normalization_39/batchnorm/ReadVariableOp_22j
3batch_normalization_39/batchnorm/mul/ReadVariableOp3batch_normalization_39/batchnorm/mul/ReadVariableOp2D
 conv1d_38/BiasAdd/ReadVariableOp conv1d_38/BiasAdd/ReadVariableOp2\
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_39/BiasAdd/ReadVariableOp conv1d_39/BiasAdd/ReadVariableOp2\
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
Ю
Њ
7__inference_batch_normalization_39_layer_call_fn_420539

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_4193642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ$::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ$
 
_user_specified_nameinputs
у
~
)__inference_dense_57_layer_call_fn_420570

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_4194262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџи::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
ч	
н
D__inference_dense_57_layer_call_and_return_conditional_losses_419426

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџи::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
ђ
Њ
7__inference_batch_normalization_39_layer_call_fn_420457

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_4191232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Еp

__inference__traced_save_420852
file_prefix/
+savev2_conv1d_38_kernel_read_readvariableop-
)savev2_conv1d_38_bias_read_readvariableop;
7savev2_batch_normalization_38_gamma_read_readvariableop:
6savev2_batch_normalization_38_beta_read_readvariableopA
=savev2_batch_normalization_38_moving_mean_read_readvariableopE
Asavev2_batch_normalization_38_moving_variance_read_readvariableop/
+savev2_conv1d_39_kernel_read_readvariableop-
)savev2_conv1d_39_bias_read_readvariableop;
7savev2_batch_normalization_39_gamma_read_readvariableop:
6savev2_batch_normalization_39_beta_read_readvariableopA
=savev2_batch_normalization_39_moving_mean_read_readvariableopE
Asavev2_batch_normalization_39_moving_variance_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop.
*savev2_dense_58_kernel_read_readvariableop,
(savev2_dense_58_bias_read_readvariableop.
*savev2_dense_59_kernel_read_readvariableop,
(savev2_dense_59_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv1d_38_kernel_m_read_readvariableop4
0savev2_adam_conv1d_38_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_38_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_38_beta_m_read_readvariableop6
2savev2_adam_conv1d_39_kernel_m_read_readvariableop4
0savev2_adam_conv1d_39_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_39_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_39_beta_m_read_readvariableop5
1savev2_adam_dense_57_kernel_m_read_readvariableop3
/savev2_adam_dense_57_bias_m_read_readvariableop5
1savev2_adam_dense_58_kernel_m_read_readvariableop3
/savev2_adam_dense_58_bias_m_read_readvariableop5
1savev2_adam_dense_59_kernel_m_read_readvariableop3
/savev2_adam_dense_59_bias_m_read_readvariableop6
2savev2_adam_conv1d_38_kernel_v_read_readvariableop4
0savev2_adam_conv1d_38_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_38_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_38_beta_v_read_readvariableop6
2savev2_adam_conv1d_39_kernel_v_read_readvariableop4
0savev2_adam_conv1d_39_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_39_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_39_beta_v_read_readvariableop5
1savev2_adam_dense_57_kernel_v_read_readvariableop3
/savev2_adam_dense_57_bias_v_read_readvariableop5
1savev2_adam_dense_58_kernel_v_read_readvariableop3
/savev2_adam_dense_58_bias_v_read_readvariableop5
1savev2_adam_dense_59_kernel_v_read_readvariableop3
/savev2_adam_dense_59_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameъ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*ќ
valueђBя8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesљ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЙ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_38_kernel_read_readvariableop)savev2_conv1d_38_bias_read_readvariableop7savev2_batch_normalization_38_gamma_read_readvariableop6savev2_batch_normalization_38_beta_read_readvariableop=savev2_batch_normalization_38_moving_mean_read_readvariableopAsavev2_batch_normalization_38_moving_variance_read_readvariableop+savev2_conv1d_39_kernel_read_readvariableop)savev2_conv1d_39_bias_read_readvariableop7savev2_batch_normalization_39_gamma_read_readvariableop6savev2_batch_normalization_39_beta_read_readvariableop=savev2_batch_normalization_39_moving_mean_read_readvariableopAsavev2_batch_normalization_39_moving_variance_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop*savev2_dense_58_kernel_read_readvariableop(savev2_dense_58_bias_read_readvariableop*savev2_dense_59_kernel_read_readvariableop(savev2_dense_59_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv1d_38_kernel_m_read_readvariableop0savev2_adam_conv1d_38_bias_m_read_readvariableop>savev2_adam_batch_normalization_38_gamma_m_read_readvariableop=savev2_adam_batch_normalization_38_beta_m_read_readvariableop2savev2_adam_conv1d_39_kernel_m_read_readvariableop0savev2_adam_conv1d_39_bias_m_read_readvariableop>savev2_adam_batch_normalization_39_gamma_m_read_readvariableop=savev2_adam_batch_normalization_39_beta_m_read_readvariableop1savev2_adam_dense_57_kernel_m_read_readvariableop/savev2_adam_dense_57_bias_m_read_readvariableop1savev2_adam_dense_58_kernel_m_read_readvariableop/savev2_adam_dense_58_bias_m_read_readvariableop1savev2_adam_dense_59_kernel_m_read_readvariableop/savev2_adam_dense_59_bias_m_read_readvariableop2savev2_adam_conv1d_38_kernel_v_read_readvariableop0savev2_adam_conv1d_38_bias_v_read_readvariableop>savev2_adam_batch_normalization_38_gamma_v_read_readvariableop=savev2_adam_batch_normalization_38_beta_v_read_readvariableop2savev2_adam_conv1d_39_kernel_v_read_readvariableop0savev2_adam_conv1d_39_bias_v_read_readvariableop>savev2_adam_batch_normalization_39_gamma_v_read_readvariableop=savev2_adam_batch_normalization_39_beta_v_read_readvariableop1savev2_adam_dense_57_kernel_v_read_readvariableop/savev2_adam_dense_57_bias_v_read_readvariableop1savev2_adam_dense_58_kernel_v_read_readvariableop/savev2_adam_dense_58_bias_v_read_readvariableop1savev2_adam_dense_59_kernel_v_read_readvariableop/savev2_adam_dense_59_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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
: :::::::::::::	и:::::: : : : : : : : : :::::::::	и::::::::::::::	и:::::: 2(
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
:	и: 
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
:	и: %
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
:	и: 3
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

e
F__inference_dropout_39_layer_call_and_return_conditional_losses_420629

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

M
1__inference_max_pooling1d_39_layer_call_fn_419149

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_39_layer_call_and_return_conditional_losses_4191432
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ј
E__inference_conv1d_38_layer_call_and_return_conditional_losses_420177

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџH2
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџH*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџH2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџH2
ReluЈ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџH2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџH::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
К:
ђ
I__inference_sequential_19_layer_call_and_return_conditional_losses_419609
conv1d_38_input
conv1d_38_419560
conv1d_38_419562!
batch_normalization_38_419565!
batch_normalization_38_419567!
batch_normalization_38_419569!
batch_normalization_38_419571
conv1d_39_419575
conv1d_39_419577!
batch_normalization_39_419580!
batch_normalization_39_419582!
batch_normalization_39_419584!
batch_normalization_39_419586
dense_57_419591
dense_57_419593
dense_58_419597
dense_58_419599
dense_59_419603
dense_59_419605
identityЂ.batch_normalization_38/StatefulPartitionedCallЂ.batch_normalization_39/StatefulPartitionedCallЂ!conv1d_38/StatefulPartitionedCallЂ!conv1d_39/StatefulPartitionedCallЂ dense_57/StatefulPartitionedCallЂ dense_58/StatefulPartitionedCallЂ dense_59/StatefulPartitionedCallЋ
!conv1d_38/StatefulPartitionedCallStatefulPartitionedCallconv1d_38_inputconv1d_38_419560conv1d_38_419562*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_38_layer_call_and_return_conditional_losses_4191692#
!conv1d_38/StatefulPartitionedCallЩ
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall*conv1d_38/StatefulPartitionedCall:output:0batch_normalization_38_419565batch_normalization_38_419567batch_normalization_38_419569batch_normalization_38_419571*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_41924020
.batch_normalization_38/StatefulPartitionedCallІ
 max_pooling1d_38/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_38_layer_call_and_return_conditional_losses_4189882"
 max_pooling1d_38/PartitionedCallХ
!conv1d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_38/PartitionedCall:output:0conv1d_39_419575conv1d_39_419577*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_39_layer_call_and_return_conditional_losses_4192932#
!conv1d_39/StatefulPartitionedCallЩ
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv1d_39/StatefulPartitionedCall:output:0batch_normalization_39_419580batch_normalization_39_419582batch_normalization_39_419584batch_normalization_39_419586*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_41936420
.batch_normalization_39/StatefulPartitionedCallІ
 max_pooling1d_39/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_39_layer_call_and_return_conditional_losses_4191432"
 max_pooling1d_39/PartitionedCall
flatten_19/PartitionedCallPartitionedCall)max_pooling1d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџи* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_4194072
flatten_19/PartitionedCallЖ
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#flatten_19/PartitionedCall:output:0dense_57_419591dense_57_419593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_4194262"
 dense_57/StatefulPartitionedCall
dropout_38/PartitionedCallPartitionedCall)dense_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_4194592
dropout_38/PartitionedCallЖ
 dense_58/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_58_419597dense_58_419599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_4194832"
 dense_58/StatefulPartitionedCall
dropout_39/PartitionedCallPartitionedCall)dense_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_4195162
dropout_39/PartitionedCallЖ
 dense_59/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0dense_59_419603dense_59_419605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_4195402"
 dense_59/StatefulPartitionedCall
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall"^conv1d_38/StatefulPartitionedCall"^conv1d_39/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2F
!conv1d_38/StatefulPartitionedCall!conv1d_38/StatefulPartitionedCall2F
!conv1d_39/StatefulPartitionedCall!conv1d_39/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:\ X
+
_output_shapes
:џџџџџџџџџH
)
_user_specified_nameconv1d_38_input
ѓ

*__inference_conv1d_38_layer_call_fn_420186

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_38_layer_call_and_return_conditional_losses_4191692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџH2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџH::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs


R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_419123

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 :џџџџџџџџџџџџџџџџџџ2
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
 :џџџџџџџџџџџџџџџџџџ2
batchnorm/add_1ш
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ш
h
L__inference_max_pooling1d_39_layer_call_and_return_conditional_losses_419143

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsА
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

њ
.__inference_sequential_19_layer_call_fn_420161

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
identityЂStatefulPartitionedCallе
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
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_4197572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs


R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420242

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 :џџџџџџџџџџџџџџџџџџ2
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
 :џџџџџџџџџџџџџџџџџџ2
batchnorm/add_1ш
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж
b
F__inference_flatten_19_layer_call_and_return_conditional_losses_420545

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџи   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџи2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџи2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў

њ
.__inference_sequential_19_layer_call_fn_420120

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
identityЂStatefulPartitionedCallб
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
:џџџџџџџџџ*0
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_4196642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
щ

R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420513

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
:џџџџџџџџџ$2
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
:џџџџџџџџџ$2
batchnorm/add_1п
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ$::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ$
 
_user_specified_nameinputs
с
~
)__inference_dense_59_layer_call_fn_420664

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_4195402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
d
+__inference_dropout_39_layer_call_fn_420639

inputs
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_4195112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
є0
Щ
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420411

inputs
assignmovingavg_420386
assignmovingavg_1_420392)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
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
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
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
moments/Squeeze_1Ь
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/420386*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_420386*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/420386*
_output_shapes
:2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/420386*
_output_shapes
:2
AssignMovingAvg/mulЏ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_420386AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/420386*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/420392*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_420392*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/420392*
_output_shapes
:2
AssignMovingAvg_1/subђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/420392*
_output_shapes
:2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_420392AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/420392*
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
 :џџџџџџџџџџџџџџџџџџ2
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
 :џџџџџџџџџџџџџџџџџџ2
batchnorm/add_1Р
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Вю
Ж
I__inference_sequential_19_layer_call_and_return_conditional_losses_419986

inputs9
5conv1d_38_conv1d_expanddims_1_readvariableop_resource-
)conv1d_38_biasadd_readvariableop_resource1
-batch_normalization_38_assignmovingavg_4198703
/batch_normalization_38_assignmovingavg_1_419876@
<batch_normalization_38_batchnorm_mul_readvariableop_resource<
8batch_normalization_38_batchnorm_readvariableop_resource9
5conv1d_39_conv1d_expanddims_1_readvariableop_resource-
)conv1d_39_biasadd_readvariableop_resource1
-batch_normalization_39_assignmovingavg_4199183
/batch_normalization_39_assignmovingavg_1_419924@
<batch_normalization_39_batchnorm_mul_readvariableop_resource<
8batch_normalization_39_batchnorm_readvariableop_resource+
'dense_57_matmul_readvariableop_resource,
(dense_57_biasadd_readvariableop_resource+
'dense_58_matmul_readvariableop_resource,
(dense_58_biasadd_readvariableop_resource+
'dense_59_matmul_readvariableop_resource,
(dense_59_biasadd_readvariableop_resource
identityЂ:batch_normalization_38/AssignMovingAvg/AssignSubVariableOpЂ5batch_normalization_38/AssignMovingAvg/ReadVariableOpЂ<batch_normalization_38/AssignMovingAvg_1/AssignSubVariableOpЂ7batch_normalization_38/AssignMovingAvg_1/ReadVariableOpЂ/batch_normalization_38/batchnorm/ReadVariableOpЂ3batch_normalization_38/batchnorm/mul/ReadVariableOpЂ:batch_normalization_39/AssignMovingAvg/AssignSubVariableOpЂ5batch_normalization_39/AssignMovingAvg/ReadVariableOpЂ<batch_normalization_39/AssignMovingAvg_1/AssignSubVariableOpЂ7batch_normalization_39/AssignMovingAvg_1/ReadVariableOpЂ/batch_normalization_39/batchnorm/ReadVariableOpЂ3batch_normalization_39/batchnorm/mul/ReadVariableOpЂ conv1d_38/BiasAdd/ReadVariableOpЂ,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpЂ conv1d_39/BiasAdd/ReadVariableOpЂ,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpЂdense_57/BiasAdd/ReadVariableOpЂdense_57/MatMul/ReadVariableOpЂdense_58/BiasAdd/ReadVariableOpЂdense_58/MatMul/ReadVariableOpЂdense_59/BiasAdd/ReadVariableOpЂdense_59/MatMul/ReadVariableOp
conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2!
conv1d_38/conv1d/ExpandDims/dimД
conv1d_38/conv1d/ExpandDims
ExpandDimsinputs(conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџH2
conv1d_38/conv1d/ExpandDimsж
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_38/conv1d/ExpandDims_1/dimп
conv1d_38/conv1d/ExpandDims_1
ExpandDims4conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_38/conv1d/ExpandDims_1о
conv1d_38/conv1dConv2D$conv1d_38/conv1d/ExpandDims:output:0&conv1d_38/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџH*
paddingSAME*
strides
2
conv1d_38/conv1dА
conv1d_38/conv1d/SqueezeSqueezeconv1d_38/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџH*
squeeze_dims

§џџџџџџџџ2
conv1d_38/conv1d/SqueezeЊ
 conv1d_38/BiasAdd/ReadVariableOpReadVariableOp)conv1d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_38/BiasAdd/ReadVariableOpД
conv1d_38/BiasAddBiasAdd!conv1d_38/conv1d/Squeeze:output:0(conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџH2
conv1d_38/BiasAddz
conv1d_38/ReluReluconv1d_38/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџH2
conv1d_38/ReluП
5batch_normalization_38/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_38/moments/mean/reduction_indicesю
#batch_normalization_38/moments/meanMeanconv1d_38/Relu:activations:0>batch_normalization_38/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_38/moments/meanХ
+batch_normalization_38/moments/StopGradientStopGradient,batch_normalization_38/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_38/moments/StopGradient
0batch_normalization_38/moments/SquaredDifferenceSquaredDifferenceconv1d_38/Relu:activations:04batch_normalization_38/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџH22
0batch_normalization_38/moments/SquaredDifferenceЧ
9batch_normalization_38/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_38/moments/variance/reduction_indices
'batch_normalization_38/moments/varianceMean4batch_normalization_38/moments/SquaredDifference:z:0Bbatch_normalization_38/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_38/moments/varianceЦ
&batch_normalization_38/moments/SqueezeSqueeze,batch_normalization_38/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_38/moments/SqueezeЮ
(batch_normalization_38/moments/Squeeze_1Squeeze0batch_normalization_38/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_38/moments/Squeeze_1
,batch_normalization_38/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_38/AssignMovingAvg/419870*
_output_shapes
: *
dtype0*
valueB
 *
з#<2.
,batch_normalization_38/AssignMovingAvg/decayи
5batch_normalization_38/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_38_assignmovingavg_419870*
_output_shapes
:*
dtype027
5batch_normalization_38/AssignMovingAvg/ReadVariableOpф
*batch_normalization_38/AssignMovingAvg/subSub=batch_normalization_38/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_38/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_38/AssignMovingAvg/419870*
_output_shapes
:2,
*batch_normalization_38/AssignMovingAvg/subл
*batch_normalization_38/AssignMovingAvg/mulMul.batch_normalization_38/AssignMovingAvg/sub:z:05batch_normalization_38/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_38/AssignMovingAvg/419870*
_output_shapes
:2,
*batch_normalization_38/AssignMovingAvg/mulЙ
:batch_normalization_38/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_38_assignmovingavg_419870.batch_normalization_38/AssignMovingAvg/mul:z:06^batch_normalization_38/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_38/AssignMovingAvg/419870*
_output_shapes
 *
dtype02<
:batch_normalization_38/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_38/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_38/AssignMovingAvg_1/419876*
_output_shapes
: *
dtype0*
valueB
 *
з#<20
.batch_normalization_38/AssignMovingAvg_1/decayо
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_38_assignmovingavg_1_419876*
_output_shapes
:*
dtype029
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOpю
,batch_normalization_38/AssignMovingAvg_1/subSub?batch_normalization_38/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_38/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_38/AssignMovingAvg_1/419876*
_output_shapes
:2.
,batch_normalization_38/AssignMovingAvg_1/subх
,batch_normalization_38/AssignMovingAvg_1/mulMul0batch_normalization_38/AssignMovingAvg_1/sub:z:07batch_normalization_38/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_38/AssignMovingAvg_1/419876*
_output_shapes
:2.
,batch_normalization_38/AssignMovingAvg_1/mulХ
<batch_normalization_38/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_38_assignmovingavg_1_4198760batch_normalization_38/AssignMovingAvg_1/mul:z:08^batch_normalization_38/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_38/AssignMovingAvg_1/419876*
_output_shapes
 *
dtype02>
<batch_normalization_38/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_38/batchnorm/add/yо
$batch_normalization_38/batchnorm/addAddV21batch_normalization_38/moments/Squeeze_1:output:0/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/addЈ
&batch_normalization_38/batchnorm/RsqrtRsqrt(batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/Rsqrtу
3batch_normalization_38/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_38_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_38/batchnorm/mul/ReadVariableOpс
$batch_normalization_38/batchnorm/mulMul*batch_normalization_38/batchnorm/Rsqrt:y:0;batch_normalization_38/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/mulе
&batch_normalization_38/batchnorm/mul_1Mulconv1d_38/Relu:activations:0(batch_normalization_38/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџH2(
&batch_normalization_38/batchnorm/mul_1з
&batch_normalization_38/batchnorm/mul_2Mul/batch_normalization_38/moments/Squeeze:output:0(batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/mul_2з
/batch_normalization_38/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_38_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_38/batchnorm/ReadVariableOpн
$batch_normalization_38/batchnorm/subSub7batch_normalization_38/batchnorm/ReadVariableOp:value:0*batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/subх
&batch_normalization_38/batchnorm/add_1AddV2*batch_normalization_38/batchnorm/mul_1:z:0(batch_normalization_38/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџH2(
&batch_normalization_38/batchnorm/add_1
max_pooling1d_38/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_38/ExpandDims/dimи
max_pooling1d_38/ExpandDims
ExpandDims*batch_normalization_38/batchnorm/add_1:z:0(max_pooling1d_38/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџH2
max_pooling1d_38/ExpandDimsб
max_pooling1d_38/MaxPoolMaxPool$max_pooling1d_38/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ$*
ksize
*
paddingSAME*
strides
2
max_pooling1d_38/MaxPoolЏ
max_pooling1d_38/SqueezeSqueeze!max_pooling1d_38/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$*
squeeze_dims
2
max_pooling1d_38/Squeeze
conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2!
conv1d_39/conv1d/ExpandDims/dimЯ
conv1d_39/conv1d/ExpandDims
ExpandDims!max_pooling1d_38/Squeeze:output:0(conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$2
conv1d_39/conv1d/ExpandDimsж
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_39_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_39/conv1d/ExpandDims_1/dimп
conv1d_39/conv1d/ExpandDims_1
ExpandDims4conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_39/conv1d/ExpandDims_1о
conv1d_39/conv1dConv2D$conv1d_39/conv1d/ExpandDims:output:0&conv1d_39/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$*
paddingSAME*
strides
2
conv1d_39/conv1dА
conv1d_39/conv1d/SqueezeSqueezeconv1d_39/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$*
squeeze_dims

§џџџџџџџџ2
conv1d_39/conv1d/SqueezeЊ
 conv1d_39/BiasAdd/ReadVariableOpReadVariableOp)conv1d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_39/BiasAdd/ReadVariableOpД
conv1d_39/BiasAddBiasAdd!conv1d_39/conv1d/Squeeze:output:0(conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ$2
conv1d_39/BiasAddz
conv1d_39/ReluReluconv1d_39/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$2
conv1d_39/ReluП
5batch_normalization_39/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_39/moments/mean/reduction_indicesю
#batch_normalization_39/moments/meanMeanconv1d_39/Relu:activations:0>batch_normalization_39/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2%
#batch_normalization_39/moments/meanХ
+batch_normalization_39/moments/StopGradientStopGradient,batch_normalization_39/moments/mean:output:0*
T0*"
_output_shapes
:2-
+batch_normalization_39/moments/StopGradient
0batch_normalization_39/moments/SquaredDifferenceSquaredDifferenceconv1d_39/Relu:activations:04batch_normalization_39/moments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$22
0batch_normalization_39/moments/SquaredDifferenceЧ
9batch_normalization_39/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_39/moments/variance/reduction_indices
'batch_normalization_39/moments/varianceMean4batch_normalization_39/moments/SquaredDifference:z:0Bbatch_normalization_39/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2)
'batch_normalization_39/moments/varianceЦ
&batch_normalization_39/moments/SqueezeSqueeze,batch_normalization_39/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_39/moments/SqueezeЮ
(batch_normalization_39/moments/Squeeze_1Squeeze0batch_normalization_39/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_39/moments/Squeeze_1
,batch_normalization_39/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_39/AssignMovingAvg/419918*
_output_shapes
: *
dtype0*
valueB
 *
з#<2.
,batch_normalization_39/AssignMovingAvg/decayи
5batch_normalization_39/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_39_assignmovingavg_419918*
_output_shapes
:*
dtype027
5batch_normalization_39/AssignMovingAvg/ReadVariableOpф
*batch_normalization_39/AssignMovingAvg/subSub=batch_normalization_39/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_39/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_39/AssignMovingAvg/419918*
_output_shapes
:2,
*batch_normalization_39/AssignMovingAvg/subл
*batch_normalization_39/AssignMovingAvg/mulMul.batch_normalization_39/AssignMovingAvg/sub:z:05batch_normalization_39/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_39/AssignMovingAvg/419918*
_output_shapes
:2,
*batch_normalization_39/AssignMovingAvg/mulЙ
:batch_normalization_39/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_39_assignmovingavg_419918.batch_normalization_39/AssignMovingAvg/mul:z:06^batch_normalization_39/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_39/AssignMovingAvg/419918*
_output_shapes
 *
dtype02<
:batch_normalization_39/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_39/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_39/AssignMovingAvg_1/419924*
_output_shapes
: *
dtype0*
valueB
 *
з#<20
.batch_normalization_39/AssignMovingAvg_1/decayо
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_39_assignmovingavg_1_419924*
_output_shapes
:*
dtype029
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOpю
,batch_normalization_39/AssignMovingAvg_1/subSub?batch_normalization_39/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_39/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_39/AssignMovingAvg_1/419924*
_output_shapes
:2.
,batch_normalization_39/AssignMovingAvg_1/subх
,batch_normalization_39/AssignMovingAvg_1/mulMul0batch_normalization_39/AssignMovingAvg_1/sub:z:07batch_normalization_39/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@batch_normalization_39/AssignMovingAvg_1/419924*
_output_shapes
:2.
,batch_normalization_39/AssignMovingAvg_1/mulХ
<batch_normalization_39/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_39_assignmovingavg_1_4199240batch_normalization_39/AssignMovingAvg_1/mul:z:08^batch_normalization_39/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*B
_class8
64loc:@batch_normalization_39/AssignMovingAvg_1/419924*
_output_shapes
 *
dtype02>
<batch_normalization_39/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_39/batchnorm/add/yо
$batch_normalization_39/batchnorm/addAddV21batch_normalization_39/moments/Squeeze_1:output:0/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/addЈ
&batch_normalization_39/batchnorm/RsqrtRsqrt(batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/Rsqrtу
3batch_normalization_39/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_39_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_39/batchnorm/mul/ReadVariableOpс
$batch_normalization_39/batchnorm/mulMul*batch_normalization_39/batchnorm/Rsqrt:y:0;batch_normalization_39/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/mulе
&batch_normalization_39/batchnorm/mul_1Mulconv1d_39/Relu:activations:0(batch_normalization_39/batchnorm/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ$2(
&batch_normalization_39/batchnorm/mul_1з
&batch_normalization_39/batchnorm/mul_2Mul/batch_normalization_39/moments/Squeeze:output:0(batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/mul_2з
/batch_normalization_39/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_39_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_39/batchnorm/ReadVariableOpн
$batch_normalization_39/batchnorm/subSub7batch_normalization_39/batchnorm/ReadVariableOp:value:0*batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/subх
&batch_normalization_39/batchnorm/add_1AddV2*batch_normalization_39/batchnorm/mul_1:z:0(batch_normalization_39/batchnorm/sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ$2(
&batch_normalization_39/batchnorm/add_1
max_pooling1d_39/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_39/ExpandDims/dimи
max_pooling1d_39/ExpandDims
ExpandDims*batch_normalization_39/batchnorm/add_1:z:0(max_pooling1d_39/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$2
max_pooling1d_39/ExpandDimsб
max_pooling1d_39/MaxPoolMaxPool$max_pooling1d_39/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
max_pooling1d_39/MaxPoolЏ
max_pooling1d_39/SqueezeSqueeze!max_pooling1d_39/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
max_pooling1d_39/Squeezeu
flatten_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџи   2
flatten_19/ConstЄ
flatten_19/ReshapeReshape!max_pooling1d_39/Squeeze:output:0flatten_19/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџи2
flatten_19/ReshapeЉ
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes
:	и*
dtype02 
dense_57/MatMul/ReadVariableOpЃ
dense_57/MatMulMatMulflatten_19/Reshape:output:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_57/MatMulЇ
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_57/BiasAdd/ReadVariableOpЅ
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_57/BiasAdds
dense_57/TanhTanhdense_57/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_57/Tanhy
dropout_38/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
dropout_38/dropout/Const
dropout_38/dropout/MulMuldense_57/Tanh:y:0!dropout_38/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_38/dropout/Mulu
dropout_38/dropout/ShapeShapedense_57/Tanh:y:0*
T0*
_output_shapes
:2
dropout_38/dropout/Shapeе
/dropout_38/dropout/random_uniform/RandomUniformRandomUniform!dropout_38/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype021
/dropout_38/dropout/random_uniform/RandomUniform
!dropout_38/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2#
!dropout_38/dropout/GreaterEqual/yъ
dropout_38/dropout/GreaterEqualGreaterEqual8dropout_38/dropout/random_uniform/RandomUniform:output:0*dropout_38/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
dropout_38/dropout/GreaterEqual 
dropout_38/dropout/CastCast#dropout_38/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_38/dropout/CastІ
dropout_38/dropout/Mul_1Muldropout_38/dropout/Mul:z:0dropout_38/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_38/dropout/Mul_1Ј
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_58/MatMul/ReadVariableOpЄ
dense_58/MatMulMatMuldropout_38/dropout/Mul_1:z:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_58/MatMulЇ
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_58/BiasAdd/ReadVariableOpЅ
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_58/BiasAdds
dense_58/TanhTanhdense_58/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_58/Tanhy
dropout_39/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
dropout_39/dropout/Const
dropout_39/dropout/MulMuldense_58/Tanh:y:0!dropout_39/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_39/dropout/Mulu
dropout_39/dropout/ShapeShapedense_58/Tanh:y:0*
T0*
_output_shapes
:2
dropout_39/dropout/Shapeе
/dropout_39/dropout/random_uniform/RandomUniformRandomUniform!dropout_39/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype021
/dropout_39/dropout/random_uniform/RandomUniform
!dropout_39/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2#
!dropout_39/dropout/GreaterEqual/yъ
dropout_39/dropout/GreaterEqualGreaterEqual8dropout_39/dropout/random_uniform/RandomUniform:output:0*dropout_39/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
dropout_39/dropout/GreaterEqual 
dropout_39/dropout/CastCast#dropout_39/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_39/dropout/CastІ
dropout_39/dropout/Mul_1Muldropout_39/dropout/Mul:z:0dropout_39/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_39/dropout/Mul_1Ј
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_59/MatMul/ReadVariableOpЄ
dense_59/MatMulMatMuldropout_39/dropout/Mul_1:z:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_59/MatMulЇ
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_59/BiasAdd/ReadVariableOpЅ
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_59/BiasAdd|
dense_59/SoftmaxSoftmaxdense_59/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_59/Softmax	
IdentityIdentitydense_59/Softmax:softmax:0;^batch_normalization_38/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_38/AssignMovingAvg/ReadVariableOp=^batch_normalization_38/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_38/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_38/batchnorm/ReadVariableOp4^batch_normalization_38/batchnorm/mul/ReadVariableOp;^batch_normalization_39/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_39/AssignMovingAvg/ReadVariableOp=^batch_normalization_39/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_39/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_39/batchnorm/ReadVariableOp4^batch_normalization_39/batchnorm/mul/ReadVariableOp!^conv1d_38/BiasAdd/ReadVariableOp-^conv1d_38/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_39/BiasAdd/ReadVariableOp-^conv1d_39/conv1d/ExpandDims_1/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::2x
:batch_normalization_38/AssignMovingAvg/AssignSubVariableOp:batch_normalization_38/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_38/AssignMovingAvg/ReadVariableOp5batch_normalization_38/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_38/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_38/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_38/batchnorm/ReadVariableOp/batch_normalization_38/batchnorm/ReadVariableOp2j
3batch_normalization_38/batchnorm/mul/ReadVariableOp3batch_normalization_38/batchnorm/mul/ReadVariableOp2x
:batch_normalization_39/AssignMovingAvg/AssignSubVariableOp:batch_normalization_39/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_39/AssignMovingAvg/ReadVariableOp5batch_normalization_39/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_39/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_39/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_39/batchnorm/ReadVariableOp/batch_normalization_39/batchnorm/ReadVariableOp2j
3batch_normalization_39/batchnorm/mul/ReadVariableOp3batch_normalization_39/batchnorm/mul/ReadVariableOp2D
 conv1d_38/BiasAdd/ReadVariableOp conv1d_38/BiasAdd/ReadVariableOp2\
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_39/BiasAdd/ReadVariableOp conv1d_39/BiasAdd/ReadVariableOp2\
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
№
Њ
7__inference_batch_normalization_39_layer_call_fn_420444

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_4190902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ш
h
L__inference_max_pooling1d_38_layer_call_and_return_conditional_losses_418988

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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsА
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Щ
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_419459

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

G
+__inference_dropout_38_layer_call_fn_420597

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_4194592
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч	
н
D__inference_dense_57_layer_call_and_return_conditional_losses_420561

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџи::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
і	
н
D__inference_dense_59_layer_call_and_return_conditional_losses_420655

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і	
н
D__inference_dense_59_layer_call_and_return_conditional_losses_419540

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

e
F__inference_dropout_38_layer_call_and_return_conditional_losses_419454

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

e
F__inference_dropout_39_layer_call_and_return_conditional_losses_419511

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUе?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
d
+__inference_dropout_38_layer_call_fn_420592

inputs
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_4194542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н0
Щ
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420304

inputs
assignmovingavg_420279
assignmovingavg_1_420285)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
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
moments/StopGradientЈ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџH2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
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
moments/Squeeze_1Ь
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/420279*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_420279*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/420279*
_output_shapes
:2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/420279*
_output_shapes
:2
AssignMovingAvg/mulЏ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_420279AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/420279*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/420285*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_420285*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/420285*
_output_shapes
:2
AssignMovingAvg_1/subђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/420285*
_output_shapes
:2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_420285AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/420285*
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
:џџџџџџџџџH2
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
:џџџџџџџџџH2
batchnorm/add_1З
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџH::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
Щ
d
F__inference_dropout_39_layer_call_and_return_conditional_losses_419516

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ж
b
F__inference_flatten_19_layer_call_and_return_conditional_losses_419407

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџи   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџи2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџи2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ

*__inference_conv1d_39_layer_call_fn_420375

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_39_layer_call_and_return_conditional_losses_4192932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ$::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ$
 
_user_specified_nameinputs
№
Њ
7__inference_batch_normalization_38_layer_call_fn_420255

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_4189352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

G
+__inference_dropout_39_layer_call_fn_420644

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_4195162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


.__inference_sequential_19_layer_call_fn_419703
conv1d_38_input
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
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallconv1d_38_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*0
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_sequential_19_layer_call_and_return_conditional_losses_4196642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:џџџџџџџџџH
)
_user_specified_nameconv1d_38_input
Ј=
Г
I__inference_sequential_19_layer_call_and_return_conditional_losses_419664

inputs
conv1d_38_419615
conv1d_38_419617!
batch_normalization_38_419620!
batch_normalization_38_419622!
batch_normalization_38_419624!
batch_normalization_38_419626
conv1d_39_419630
conv1d_39_419632!
batch_normalization_39_419635!
batch_normalization_39_419637!
batch_normalization_39_419639!
batch_normalization_39_419641
dense_57_419646
dense_57_419648
dense_58_419652
dense_58_419654
dense_59_419658
dense_59_419660
identityЂ.batch_normalization_38/StatefulPartitionedCallЂ.batch_normalization_39/StatefulPartitionedCallЂ!conv1d_38/StatefulPartitionedCallЂ!conv1d_39/StatefulPartitionedCallЂ dense_57/StatefulPartitionedCallЂ dense_58/StatefulPartitionedCallЂ dense_59/StatefulPartitionedCallЂ"dropout_38/StatefulPartitionedCallЂ"dropout_39/StatefulPartitionedCallЂ
!conv1d_38/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_38_419615conv1d_38_419617*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_38_layer_call_and_return_conditional_losses_4191692#
!conv1d_38/StatefulPartitionedCallЧ
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall*conv1d_38/StatefulPartitionedCall:output:0batch_normalization_38_419620batch_normalization_38_419622batch_normalization_38_419624batch_normalization_38_419626*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_41922020
.batch_normalization_38/StatefulPartitionedCallІ
 max_pooling1d_38/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_38_layer_call_and_return_conditional_losses_4189882"
 max_pooling1d_38/PartitionedCallХ
!conv1d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_38/PartitionedCall:output:0conv1d_39_419630conv1d_39_419632*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv1d_39_layer_call_and_return_conditional_losses_4192932#
!conv1d_39/StatefulPartitionedCallЧ
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv1d_39/StatefulPartitionedCall:output:0batch_normalization_39_419635batch_normalization_39_419637batch_normalization_39_419639batch_normalization_39_419641*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_41934420
.batch_normalization_39/StatefulPartitionedCallІ
 max_pooling1d_39/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling1d_39_layer_call_and_return_conditional_losses_4191432"
 max_pooling1d_39/PartitionedCall
flatten_19/PartitionedCallPartitionedCall)max_pooling1d_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџи* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_4194072
flatten_19/PartitionedCallЖ
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#flatten_19/PartitionedCall:output:0dense_57_419646dense_57_419648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_4194262"
 dense_57/StatefulPartitionedCall
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_4194542$
"dropout_38/StatefulPartitionedCallО
 dense_58/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_58_419652dense_58_419654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_4194832"
 dense_58/StatefulPartitionedCallП
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0#^dropout_38/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_4195112$
"dropout_39/StatefulPartitionedCallО
 dense_59/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0dense_59_419658dense_59_419660*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_4195402"
 dense_59/StatefulPartitionedCallк
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall"^conv1d_38/StatefulPartitionedCall"^conv1d_39/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:џџџџџџџџџH::::::::::::::::::2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2F
!conv1d_38/StatefulPartitionedCall!conv1d_38/StatefulPartitionedCall2F
!conv1d_39/StatefulPartitionedCall!conv1d_39/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
щ

R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_419240

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
:џџџџџџџџџH2
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
:џџџџџџџџџH2
batchnorm/add_1п
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџH2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџH::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs


R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_418968

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 :џџџџџџџџџџџџџџџџџџ2
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
 :џџџџџџџџџџџџџџџџџџ2
batchnorm/add_1ш
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ј
E__inference_conv1d_39_layer_call_and_return_conditional_losses_420366

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$2
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ$2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$2
ReluЈ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ$
 
_user_specified_nameinputs
ђ
Њ
7__inference_batch_normalization_38_layer_call_fn_420268

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_4189682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:џџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ь
Њ
7__inference_batch_normalization_39_layer_call_fn_420526

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_4193442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ$::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ$
 
_user_specified_nameinputs
ущ

"__inference__traced_restore_421027
file_prefix%
!assignvariableop_conv1d_38_kernel%
!assignvariableop_1_conv1d_38_bias3
/assignvariableop_2_batch_normalization_38_gamma2
.assignvariableop_3_batch_normalization_38_beta9
5assignvariableop_4_batch_normalization_38_moving_mean=
9assignvariableop_5_batch_normalization_38_moving_variance'
#assignvariableop_6_conv1d_39_kernel%
!assignvariableop_7_conv1d_39_bias3
/assignvariableop_8_batch_normalization_39_gamma2
.assignvariableop_9_batch_normalization_39_beta:
6assignvariableop_10_batch_normalization_39_moving_mean>
:assignvariableop_11_batch_normalization_39_moving_variance'
#assignvariableop_12_dense_57_kernel%
!assignvariableop_13_dense_57_bias'
#assignvariableop_14_dense_58_kernel%
!assignvariableop_15_dense_58_bias'
#assignvariableop_16_dense_59_kernel%
!assignvariableop_17_dense_59_bias
assignvariableop_18_beta_1
assignvariableop_19_beta_2
assignvariableop_20_decay%
!assignvariableop_21_learning_rate!
assignvariableop_22_adam_iter
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1/
+assignvariableop_27_adam_conv1d_38_kernel_m-
)assignvariableop_28_adam_conv1d_38_bias_m;
7assignvariableop_29_adam_batch_normalization_38_gamma_m:
6assignvariableop_30_adam_batch_normalization_38_beta_m/
+assignvariableop_31_adam_conv1d_39_kernel_m-
)assignvariableop_32_adam_conv1d_39_bias_m;
7assignvariableop_33_adam_batch_normalization_39_gamma_m:
6assignvariableop_34_adam_batch_normalization_39_beta_m.
*assignvariableop_35_adam_dense_57_kernel_m,
(assignvariableop_36_adam_dense_57_bias_m.
*assignvariableop_37_adam_dense_58_kernel_m,
(assignvariableop_38_adam_dense_58_bias_m.
*assignvariableop_39_adam_dense_59_kernel_m,
(assignvariableop_40_adam_dense_59_bias_m/
+assignvariableop_41_adam_conv1d_38_kernel_v-
)assignvariableop_42_adam_conv1d_38_bias_v;
7assignvariableop_43_adam_batch_normalization_38_gamma_v:
6assignvariableop_44_adam_batch_normalization_38_beta_v/
+assignvariableop_45_adam_conv1d_39_kernel_v-
)assignvariableop_46_adam_conv1d_39_bias_v;
7assignvariableop_47_adam_batch_normalization_39_gamma_v:
6assignvariableop_48_adam_batch_normalization_39_beta_v.
*assignvariableop_49_adam_dense_57_kernel_v,
(assignvariableop_50_adam_dense_57_bias_v.
*assignvariableop_51_adam_dense_58_kernel_v,
(assignvariableop_52_adam_dense_58_bias_v.
*assignvariableop_53_adam_dense_59_kernel_v,
(assignvariableop_54_adam_dense_59_bias_v
identity_56ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9№
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*ќ
valueђBя8B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesџ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЦ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*і
_output_shapesу
р::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_38_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_38_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Д
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_38_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Г
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_38_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4К
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_38_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5О
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_38_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ј
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv1d_39_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7І
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv1d_39_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Д
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_39_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Г
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_39_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10О
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_39_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Т
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_39_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ћ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_57_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Љ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_57_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ћ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_58_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Љ
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_58_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ћ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_59_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Љ
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_59_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ђ
AssignVariableOp_18AssignVariableOpassignvariableop_18_beta_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ђ
AssignVariableOp_19AssignVariableOpassignvariableop_19_beta_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ё
AssignVariableOp_20AssignVariableOpassignvariableop_20_decayIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Љ
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22Ѕ
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ё
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ё
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ѓ
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ѓ
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Г
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv1d_38_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Б
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv1d_38_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29П
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_batch_normalization_38_gamma_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30О
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_batch_normalization_38_beta_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Г
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv1d_39_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Б
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv1d_39_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33П
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_batch_normalization_39_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34О
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_batch_normalization_39_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35В
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_57_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36А
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_57_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37В
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_58_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38А
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_58_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39В
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_59_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40А
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_59_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Г
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv1d_38_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Б
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv1d_38_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43П
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_38_gamma_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44О
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_38_beta_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Г
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv1d_39_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Б
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv1d_39_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47П
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_batch_normalization_39_gamma_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48О
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_batch_normalization_39_beta_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49В
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_57_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50А
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_57_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51В
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_58_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52А
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_58_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53В
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_59_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54А
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_59_bias_vIdentity_54:output:0"/device:CPU:0*
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
identity_56Identity_56:output:0*ѓ
_input_shapesс
о: :::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
Ѕ
G
+__inference_flatten_19_layer_call_fn_420550

inputs
identityЪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџи* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_19_layer_call_and_return_conditional_losses_4194072
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџи2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н0
Щ
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_419344

inputs
assignmovingavg_419319
assignmovingavg_1_419325)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
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
moments/StopGradientЈ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
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
moments/Squeeze_1Ь
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/419319*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_419319*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/419319*
_output_shapes
:2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/419319*
_output_shapes
:2
AssignMovingAvg/mulЏ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_419319AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/419319*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpв
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/419325*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_419325*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/419325*
_output_shapes
:2
AssignMovingAvg_1/subђ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/419325*
_output_shapes
:2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_419325AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/419325*
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
:џџџџџџџџџ$2
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
:џџџџџџџџџ$2
batchnorm/add_1З
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ$::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ$
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*П
serving_defaultЋ
O
conv1d_38_input<
!serving_default_conv1d_38_input:0џџџџџџџџџH<
dense_590
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ю
жZ
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
Ф_default_save_signature
Х__call__
+Ц&call_and_return_all_conditional_losses"ДV
_tf_keras_sequentialV{"class_name": "Sequential", "name": "sequential_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_38_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_38", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_38", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_38", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_39", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_39", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_39", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_19", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_38", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_39", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_38_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_38", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_38", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_38", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_39", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_39", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_39", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_19", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_38", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_39", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
р


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"Й	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_38", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72, 1]}, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 1]}}
К	
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
 	variables
!	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"ф
_tf_keras_layerЪ{"class_name": "BatchNormalization", "name": "batch_normalization_38", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_38", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72, 12]}}
ќ
"trainable_variables
#regularization_losses
$	variables
%	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"ы
_tf_keras_layerб{"class_name": "MaxPooling1D", "name": "max_pooling1d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_38", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
щ	

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"Т
_tf_keras_layerЈ{"class_name": "Conv1D", "name": "conv1d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_39", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 12]}}
К	
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1trainable_variables
2regularization_losses
3	variables
4	keras_api
Я__call__
+а&call_and_return_all_conditional_losses"ф
_tf_keras_layerЪ{"class_name": "BatchNormalization", "name": "batch_normalization_39", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_39", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 12]}}
ќ
5trainable_variables
6regularization_losses
7	variables
8	keras_api
б__call__
+в&call_and_return_all_conditional_losses"ы
_tf_keras_layerб{"class_name": "MaxPooling1D", "name": "max_pooling1d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_39", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ъ
9trainable_variables
:regularization_losses
;	variables
<	keras_api
г__call__
+д&call_and_return_all_conditional_losses"й
_tf_keras_layerП{"class_name": "Flatten", "name": "flatten_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_19", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ѕ

=kernel
>bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"Ю
_tf_keras_layerД{"class_name": "Dense", "name": "dense_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 216}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 216]}}
щ
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
з__call__
+и&call_and_return_all_conditional_losses"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_38", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_38", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
ё

Gkernel
Hbias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
й__call__
+к&call_and_return_all_conditional_losses"Ъ
_tf_keras_layerА{"class_name": "Dense", "name": "dense_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
щ
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
л__call__
+м&call_and_return_all_conditional_losses"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_39", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_39", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
є

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
н__call__
+о&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_59", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
ы

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate
[itermЈmЉmЊmЋ&mЌ'm­-mЎ.mЏ=mА>mБGmВHmГQmДRmЕvЖvЗvИvЙ&vК'vЛ-vМ.vН=vО>vПGvРHvСQvТRvУ"
	optimizer
 "
trackable_list_wrapper
І
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
Ю
\layer_regularization_losses
regularization_losses

]layers
^non_trainable_variables
	variables
_layer_metrics
trainable_variables
`metrics
Х__call__
Ф_default_save_signature
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
-
пserving_default"
signature_map
&:$2conv1d_38/kernel
:2conv1d_38/bias
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
А
trainable_variables
alayer_regularization_losses
regularization_losses

blayers
	variables
clayer_metrics
dnon_trainable_variables
emetrics
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_38/gamma
):'2batch_normalization_38/beta
2:0 (2"batch_normalization_38/moving_mean
6:4 (2&batch_normalization_38/moving_variance
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
А
trainable_variables
flayer_regularization_losses
regularization_losses

glayers
 	variables
hlayer_metrics
inon_trainable_variables
jmetrics
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
"trainable_variables
klayer_regularization_losses
#regularization_losses

llayers
$	variables
mlayer_metrics
nnon_trainable_variables
ometrics
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
&:$2conv1d_39/kernel
:2conv1d_39/bias
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
А
(trainable_variables
player_regularization_losses
)regularization_losses

qlayers
*	variables
rlayer_metrics
snon_trainable_variables
tmetrics
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_39/gamma
):'2batch_normalization_39/beta
2:0 (2"batch_normalization_39/moving_mean
6:4 (2&batch_normalization_39/moving_variance
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
А
1trainable_variables
ulayer_regularization_losses
2regularization_losses

vlayers
3	variables
wlayer_metrics
xnon_trainable_variables
ymetrics
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
5trainable_variables
zlayer_regularization_losses
6regularization_losses

{layers
7	variables
|layer_metrics
}non_trainable_variables
~metrics
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Д
9trainable_variables
layer_regularization_losses
:regularization_losses
layers
;	variables
layer_metrics
non_trainable_variables
metrics
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
": 	и2dense_57/kernel
:2dense_57/bias
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
Е
?trainable_variables
 layer_regularization_losses
@regularization_losses
layers
A	variables
layer_metrics
non_trainable_variables
metrics
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ctrainable_variables
 layer_regularization_losses
Dregularization_losses
layers
E	variables
layer_metrics
non_trainable_variables
metrics
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
!:2dense_58/kernel
:2dense_58/bias
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
Е
Itrainable_variables
 layer_regularization_losses
Jregularization_losses
layers
K	variables
layer_metrics
non_trainable_variables
metrics
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Mtrainable_variables
 layer_regularization_losses
Nregularization_losses
layers
O	variables
layer_metrics
non_trainable_variables
metrics
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
!:2dense_59/kernel
:2dense_59/bias
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
Е
Strainable_variables
 layer_regularization_losses
Tregularization_losses
layers
U	variables
layer_metrics
non_trainable_variables
metrics
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
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
П

total

 count
Ё	variables
Ђ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


Ѓtotal

Єcount
Ѕ
_fn_kwargs
І	variables
Ї	keras_api"И
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
0
 1"
trackable_list_wrapper
.
Ё	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
.
І	variables"
_generic_user_object
+:)2Adam/conv1d_38/kernel/m
!:2Adam/conv1d_38/bias/m
/:-2#Adam/batch_normalization_38/gamma/m
.:,2"Adam/batch_normalization_38/beta/m
+:)2Adam/conv1d_39/kernel/m
!:2Adam/conv1d_39/bias/m
/:-2#Adam/batch_normalization_39/gamma/m
.:,2"Adam/batch_normalization_39/beta/m
':%	и2Adam/dense_57/kernel/m
 :2Adam/dense_57/bias/m
&:$2Adam/dense_58/kernel/m
 :2Adam/dense_58/bias/m
&:$2Adam/dense_59/kernel/m
 :2Adam/dense_59/bias/m
+:)2Adam/conv1d_38/kernel/v
!:2Adam/conv1d_38/bias/v
/:-2#Adam/batch_normalization_38/gamma/v
.:,2"Adam/batch_normalization_38/beta/v
+:)2Adam/conv1d_39/kernel/v
!:2Adam/conv1d_39/bias/v
/:-2#Adam/batch_normalization_39/gamma/v
.:,2"Adam/batch_normalization_39/beta/v
':%	и2Adam/dense_57/kernel/v
 :2Adam/dense_57/bias/v
&:$2Adam/dense_58/kernel/v
 :2Adam/dense_58/bias/v
&:$2Adam/dense_59/kernel/v
 :2Adam/dense_59/bias/v
ы2ш
!__inference__wrapped_model_418839Т
В
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
annotationsЊ *2Ђ/
-*
conv1d_38_inputџџџџџџџџџH
2
.__inference_sequential_19_layer_call_fn_419703
.__inference_sequential_19_layer_call_fn_420161
.__inference_sequential_19_layer_call_fn_420120
.__inference_sequential_19_layer_call_fn_419796Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
I__inference_sequential_19_layer_call_and_return_conditional_losses_420079
I__inference_sequential_19_layer_call_and_return_conditional_losses_419557
I__inference_sequential_19_layer_call_and_return_conditional_losses_419609
I__inference_sequential_19_layer_call_and_return_conditional_losses_419986Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_conv1d_38_layer_call_fn_420186Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv1d_38_layer_call_and_return_conditional_losses_420177Ђ
В
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
annotationsЊ *
 
2
7__inference_batch_normalization_38_layer_call_fn_420268
7__inference_batch_normalization_38_layer_call_fn_420337
7__inference_batch_normalization_38_layer_call_fn_420350
7__inference_batch_normalization_38_layer_call_fn_420255Д
ЋВЇ
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
kwonlydefaultsЊ 
annotationsЊ *
 
2
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420242
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420304
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420222
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420324Д
ЋВЇ
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
kwonlydefaultsЊ 
annotationsЊ *
 
2
1__inference_max_pooling1d_38_layer_call_fn_418994г
В
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
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
L__inference_max_pooling1d_38_layer_call_and_return_conditional_losses_418988г
В
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
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv1d_39_layer_call_fn_420375Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv1d_39_layer_call_and_return_conditional_losses_420366Ђ
В
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
annotationsЊ *
 
2
7__inference_batch_normalization_39_layer_call_fn_420526
7__inference_batch_normalization_39_layer_call_fn_420457
7__inference_batch_normalization_39_layer_call_fn_420539
7__inference_batch_normalization_39_layer_call_fn_420444Д
ЋВЇ
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
kwonlydefaultsЊ 
annotationsЊ *
 
2
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420411
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420493
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420431
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420513Д
ЋВЇ
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
kwonlydefaultsЊ 
annotationsЊ *
 
2
1__inference_max_pooling1d_39_layer_call_fn_419149г
В
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
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ї2Є
L__inference_max_pooling1d_39_layer_call_and_return_conditional_losses_419143г
В
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
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
е2в
+__inference_flatten_19_layer_call_fn_420550Ђ
В
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
annotationsЊ *
 
№2э
F__inference_flatten_19_layer_call_and_return_conditional_losses_420545Ђ
В
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
annotationsЊ *
 
г2а
)__inference_dense_57_layer_call_fn_420570Ђ
В
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
annotationsЊ *
 
ю2ы
D__inference_dense_57_layer_call_and_return_conditional_losses_420561Ђ
В
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
annotationsЊ *
 
2
+__inference_dropout_38_layer_call_fn_420592
+__inference_dropout_38_layer_call_fn_420597Д
ЋВЇ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ъ2Ч
F__inference_dropout_38_layer_call_and_return_conditional_losses_420587
F__inference_dropout_38_layer_call_and_return_conditional_losses_420582Д
ЋВЇ
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
kwonlydefaultsЊ 
annotationsЊ *
 
г2а
)__inference_dense_58_layer_call_fn_420617Ђ
В
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
annotationsЊ *
 
ю2ы
D__inference_dense_58_layer_call_and_return_conditional_losses_420608Ђ
В
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
annotationsЊ *
 
2
+__inference_dropout_39_layer_call_fn_420644
+__inference_dropout_39_layer_call_fn_420639Д
ЋВЇ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ъ2Ч
F__inference_dropout_39_layer_call_and_return_conditional_losses_420629
F__inference_dropout_39_layer_call_and_return_conditional_losses_420634Д
ЋВЇ
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
kwonlydefaultsЊ 
annotationsЊ *
 
г2а
)__inference_dense_59_layer_call_fn_420664Ђ
В
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
annotationsЊ *
 
ю2ы
D__inference_dense_59_layer_call_and_return_conditional_losses_420655Ђ
В
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
annotationsЊ *
 
гBа
$__inference_signature_wrapper_419847conv1d_38_input"
В
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
annotationsЊ *
 ­
!__inference__wrapped_model_418839&'0-/.=>GHQR<Ђ9
2Ђ/
-*
conv1d_38_inputџџџџџџџџџH
Њ "3Њ0
.
dense_59"
dense_59џџџџџџџџџв
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420222|@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ
p
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 в
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420242|@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ
p 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 Р
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420304j7Ђ4
-Ђ*
$!
inputsџџџџџџџџџH
p
Њ ")Ђ&

0џџџџџџџџџH
 Р
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_420324j7Ђ4
-Ђ*
$!
inputsџџџџџџџџџH
p 
Њ ")Ђ&

0џџџџџџџџџH
 Њ
7__inference_batch_normalization_38_layer_call_fn_420255o@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ
p
Њ "%"џџџџџџџџџџџџџџџџџџЊ
7__inference_batch_normalization_38_layer_call_fn_420268o@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ
p 
Њ "%"џџџџџџџџџџџџџџџџџџ
7__inference_batch_normalization_38_layer_call_fn_420337]7Ђ4
-Ђ*
$!
inputsџџџџџџџџџH
p
Њ "џџџџџџџџџH
7__inference_batch_normalization_38_layer_call_fn_420350]7Ђ4
-Ђ*
$!
inputsџџџџџџџџџH
p 
Њ "џџџџџџџџџHв
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420411|/0-.@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ
p
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 в
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420431|0-/.@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ
p 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 Р
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420493j/0-.7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ$
p
Њ ")Ђ&

0џџџџџџџџџ$
 Р
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_420513j0-/.7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ$
p 
Њ ")Ђ&

0џџџџџџџџџ$
 Њ
7__inference_batch_normalization_39_layer_call_fn_420444o/0-.@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ
p
Њ "%"џџџџџџџџџџџџџџџџџџЊ
7__inference_batch_normalization_39_layer_call_fn_420457o0-/.@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ
p 
Њ "%"џџџџџџџџџџџџџџџџџџ
7__inference_batch_normalization_39_layer_call_fn_420526]/0-.7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ$
p
Њ "џџџџџџџџџ$
7__inference_batch_normalization_39_layer_call_fn_420539]0-/.7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ$
p 
Њ "џџџџџџџџџ$­
E__inference_conv1d_38_layer_call_and_return_conditional_losses_420177d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџH
Њ ")Ђ&

0џџџџџџџџџH
 
*__inference_conv1d_38_layer_call_fn_420186W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџH
Њ "џџџџџџџџџH­
E__inference_conv1d_39_layer_call_and_return_conditional_losses_420366d&'3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ$
Њ ")Ђ&

0џџџџџџџџџ$
 
*__inference_conv1d_39_layer_call_fn_420375W&'3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ$
Њ "џџџџџџџџџ$Ѕ
D__inference_dense_57_layer_call_and_return_conditional_losses_420561]=>0Ђ-
&Ђ#
!
inputsџџџџџџџџџи
Њ "%Ђ"

0џџџџџџџџџ
 }
)__inference_dense_57_layer_call_fn_420570P=>0Ђ-
&Ђ#
!
inputsџџџџџџџџџи
Њ "џџџџџџџџџЄ
D__inference_dense_58_layer_call_and_return_conditional_losses_420608\GH/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dense_58_layer_call_fn_420617OGH/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЄ
D__inference_dense_59_layer_call_and_return_conditional_losses_420655\QR/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dense_59_layer_call_fn_420664OQR/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
F__inference_dropout_38_layer_call_and_return_conditional_losses_420582\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 І
F__inference_dropout_38_layer_call_and_return_conditional_losses_420587\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dropout_38_layer_call_fn_420592O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ~
+__inference_dropout_38_layer_call_fn_420597O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџІ
F__inference_dropout_39_layer_call_and_return_conditional_losses_420629\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 І
F__inference_dropout_39_layer_call_and_return_conditional_losses_420634\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dropout_39_layer_call_fn_420639O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ~
+__inference_dropout_39_layer_call_fn_420644O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџЇ
F__inference_flatten_19_layer_call_and_return_conditional_losses_420545]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџи
 
+__inference_flatten_19_layer_call_fn_420550P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџие
L__inference_max_pooling1d_38_layer_call_and_return_conditional_losses_418988EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ќ
1__inference_max_pooling1d_38_layer_call_fn_418994wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџе
L__inference_max_pooling1d_39_layer_call_and_return_conditional_losses_419143EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ќ
1__inference_max_pooling1d_39_layer_call_fn_419149wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџЯ
I__inference_sequential_19_layer_call_and_return_conditional_losses_419557&'/0-.=>GHQRDЂA
:Ђ7
-*
conv1d_38_inputџџџџџџџџџH
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Я
I__inference_sequential_19_layer_call_and_return_conditional_losses_419609&'0-/.=>GHQRDЂA
:Ђ7
-*
conv1d_38_inputџџџџџџџџџH
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Х
I__inference_sequential_19_layer_call_and_return_conditional_losses_419986x&'/0-.=>GHQR;Ђ8
1Ђ.
$!
inputsџџџџџџџџџH
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Х
I__inference_sequential_19_layer_call_and_return_conditional_losses_420079x&'0-/.=>GHQR;Ђ8
1Ђ.
$!
inputsџџџџџџџџџH
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 І
.__inference_sequential_19_layer_call_fn_419703t&'/0-.=>GHQRDЂA
:Ђ7
-*
conv1d_38_inputџџџџџџџџџH
p

 
Њ "џџџџџџџџџІ
.__inference_sequential_19_layer_call_fn_419796t&'0-/.=>GHQRDЂA
:Ђ7
-*
conv1d_38_inputџџџџџџџџџH
p 

 
Њ "џџџџџџџџџ
.__inference_sequential_19_layer_call_fn_420120k&'/0-.=>GHQR;Ђ8
1Ђ.
$!
inputsџџџџџџџџџH
p

 
Њ "џџџџџџџџџ
.__inference_sequential_19_layer_call_fn_420161k&'0-/.=>GHQR;Ђ8
1Ђ.
$!
inputsџџџџџџџџџH
p 

 
Њ "џџџџџџџџџУ
$__inference_signature_wrapper_419847&'0-/.=>GHQROЂL
Ђ 
EЊB
@
conv1d_38_input-*
conv1d_38_inputџџџџџџџџџH"3Њ0
.
dense_59"
dense_59џџџџџџџџџ