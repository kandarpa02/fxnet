# These are done
===== Arithmetic (binary) =====
add
sub
mul
div
pow
matmul

===== Arithmetic (unary) =====
neg

===== Elementwise math ops =====
exp
log
sin
cos
tanh

===== Reductions =====
sum
mean
max
min
prod
===== Shape / view ops =====
reshape / view
permute / transpose (via permute)

===== Indexing =====
getitem (slicing / indexing)

===== Comparison / boolean (non-diff but graph-safe) =====
greater
greater_equal
less
less_equal
equal
not_equal

where
squeeze
unsqueeze