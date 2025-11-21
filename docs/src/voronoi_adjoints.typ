// physics macros
// https://typst.app/universe/package/physica/
#import "@preview/physica:0.9.4": * 

// annotate and highlight 
// https://typst.app/universe/package/pinit/
#import "@preview/pinit:0.2.2": *

// curves, lines with arrows
#import "@preview/tiptoe:0.2.0": *

// annotated arrows
#import "@preview/fletcher:0.5.4" as fletcher: diagram, node, edge

#import "@preview/use-tabler-icons:0.8.0": *

#let toc = outline.with(depth: 1)

#let tiny(x) = text(x, size:12pt)
#let small(x) = text(x, size:16pt)
#let big(x) = text(x, size:22pt)
#let huge(x) = text(x, size:32pt)

#let centered(x) = align(center)[#x]
#let contents_slide() = slide(grid(rows: (1fr, auto, 1fr), [], toc(), []))

#let mapsto = $arrow.r.bar$

#let avec = vb[a]
#let bvec = vb[b]
#let fvec = vb[f]
#let ivec = vb[i]
#let jvec = vb[j]
#let kvec = vb[k]
#let gvec = vb[g]
#let nvec = vb[n]
#let xvec = vb[x]
#let qvec = vb[q]
#let Fvec = vb[F]
#let Jvec = vb[J]
#let Svec = vb[S]
#let uvec = vb[u]
#let Uplanet = $uvec_"planet"$
#let vvec = vb[v]
#let Uabs = $vvec_a$
#let Zvec = vb[Z]
#let zetavec = vb($zeta$)
#let Omegavec = vb($Omega$)
#let stressvec = vb($tau$)
#let windstress = $stressvec^"wind"$

#let vol = $upsilon$
#let salt = $r_s$
#let saltref = $r_0$
#let Lag = $ell$

#let KH = $(uvec_H dprod uvec_H)/2$
#let KE = $(uvec dprod uvec)/2$
#let PV = $q_"SV"$ // Saint-Venant PV
#let QGPV = $q_"QG"$ // QG PV
#let ErtelPV = $q_"Ertel"$

#let spc = h(10mm)
#let Rarrow = $arrow.r.double.long$
#let Larrow = $arrow.l.double.long$
#let LRarrow = $arrow.l.r.double.long$
#let UDarrow = $arrow.b.t.double$
#let Darrow = $arrow.b.double$
#let Uarrow = $arrow.t.double$

#let ddot(x) = $dot.double(#x)$
#let DDt(x) = $dv(#x, t, d:upright(D))$
#let ddt(x) = $dv(#x, t)$
#let ddt2(x) = $dv(#x, t, 2)$
#let Dcov(u: uvec) = $cal(D)_#u^"cov"$
#let Dvec(u: uvec) = $cal(D)_#u^"vec"$
#let Dscal(u: uvec) = $cal(D)_#u^"scal"$
#let Ddens(u: uvec) = $cal(D)_#u^"dens"$

#let d3x = $dd(xvec, 3)$
#let d2x = $dd(xvec, 2)$
#let d2S = $dd(#vb[S], 2)$

#let ll = $lt.double$
#let gg = $gt.double$
#let simeq = $tilde.eq$
#let cst = $c s t $
#let oint = $integral.cont$
#let km = $k m$
#let cm = $c m$

#let bleu(x) = text(blue)[#x]
#let rouge(x) = text(red)[#x]
#let vert(x) = text(green)[#x]
#let blanc(x) = text(white)[#x]

#let mbleu(x) = text(blue)[$#x$]
#let norm2(x) = $norm(#x)^2$

#let redbox(..args, x) = box(rouge(x), stroke:red, inset:3mm, ..args)
#let bluebox(..args, x) = box(bleu(x), stroke:blue, inset:3mm, ..args)
#let blackbox(..args, x) = box(x, stroke:black, inset:3mm, ..args)
#let shadowbox(..args, x) = shadowed(box(x, ..args), inset:3mm)
#let whitebox(..args, x) = box(x, inset:3mm, ..args)

#let high(x) = emph(x)

#let rbrace(x) = $ lr( #stack($ #x $) #h(1mm) mid("}") ) $
#let lbrace(x) = $ lr( mid("{") #h(1mm) #stack($ #x $) ) $

#let crop(img, dx, dy, ..args) = crop-image(read("../sequences/" + img, encoding:none), dx, dy, fit:"stretch", ..args)

#let v5 = v(5mm)

#let Avg = expval
#let avg(x) = $overline(#x)$
#let Dom = $cal(D)$

#let mbox(content, fill:none, stroke:red, padding:5pt) = rect(fill:fill, stroke:stroke, inset:padding)[$ #content $]

#let Fup=$F_arrow.t$
#let Fdown=$F_arrow.b$
#let Fconv=$F_arrow.t^"conv"$

#let hvap = $h^v$
#let hdry = $h^d$
#let qvap = $q_v$
#let qdry = $q_d$

#let hliq = $h^l$
#let hsalt = $h^S$
#let qliq = $q_l$
#let qsalt = $q_S$

#let Sphere = $cal(S)$
#let dSphere = $dif^2 xvec$

= Recap

#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Forward*], [*Backward*],
  ),
  // curl
  $ zeta^v = (curl uvec)^v = sum_(e in E(v)) n_(v e)uvec_e $,
  $ partial uvec^e = rouge((-nabla^perp partial zeta)^e & = -sum_(v in V(e)) n_(e v) partial zeta_v )$,
  // divergence
  $ d^i = (div uvec)^i = sum_e n_(i e) uvec^e $,
  $ partial uvec_e= rouge( -(grad partial d)_e 
    = - nabla_e partial d_i) $,
  // gradient
  $ uvec_e = (grad a)_e = sum_i n_(e i) a_i $,
  $ partial a^i = rouge(-(div partial uvec)^i =
          - sum_e n_(i e) partial uvec^e) $,
  // primal => dual
  $ h^v = sum_(i in C(v)) cal(A)^(v i) h_i $,
  $ partial h^i = rouge( sum_(v in V(i)) cal(A)^(i v) partial h_v) $,
  [],[],
  // Squared covector
  $ K^i = (uvec dot uvec)^i = 1/2 sum_(e in E(i)) l_e/d_e uvec_e^2 $,
  $ 
    partial uvec^e = (2 partial K thin uvec )^e = rouge( ( partial K_i)_e med l_e/d_e uvec_e ) \
  $,
  [],[],
  // Centered flux
  $ Fvec^e = (h uvec)^e =  1/2 sum_(i in C(e))h_i med l_e/d_e uvec_e $,
  $ 
    partial uvec^e = (h partial Fvec)^e = rouge( 1/2 sum_(i in C(e))h_i med l_e/d_e partial Fvec_e ) \
    partial h^i = (uvec dot partial Fvec)^i = rouge( 1/2 sum_(e in E(i)) l_e/d_e u_e partial Fvec_e ) \

  $,
  // TRiSK
  $ V_e = (q U^perp)_e
    = 1/2 sum_(e') w_(e e')(q_e+q_e')rouge(U^(e')) $,
  $ 
  partial U_e = & (-q partial V^perp)_e = rouge(-1/2 sum_(e') w_(e e') (q_e+q_e') partial V^(e') ), \
  partial q^e =  & (U times partial V)^e = bleu( 1/2 sum_(e') w_(e e')(partial V^e  U^(e') - U^e partial V^(e'))) $,
  // centered flux divergence
  $ d^i = div (q uvec)^i = sum_(e in 
 E(i)) n_(i e) (q_i)_e uvec^e $, 
 $
   partial uvec_e = -(q grad partial d)_e =& rouge( -(q_i)_e nabla_e partial d_i), \
   partial q^i = - (uvec dot grad partial d)^i =& bleu( -sum_(e in E(i)) uvec^e nabla_e partial d_i )
 $, 
  // multiplied gradient
  $ uvec_e = (theta grad pi)_e = (rouge(theta_i))_e nabla_e bleu(pi_i) $,
  $ 
  partial theta^i = (partial uvec dot grad pi)^i = & rouge( sum_(e in E(i)) partial uvec^e nabla_e pi_i ), \
  partial pi^i = - (div theta uvec)^i = & bleu(-sum_(e in E(i)) n_(i e) (theta_i)_e partial uvec^e )
  $,
 )
 #table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Forward*], [*Backward*],
  ),
  // divergence
  $ d_i = (div uvec)_i = 1/cal(A)^i sum_e n_(i e) uvec^e $,
  $ partial uvec_e= rouge( -(grad partial d)_e 
    = - nabla_e (partial d^i) / cal(A)^i) $,
  // Squared covector
  $ K_i = (uvec dot uvec)_i = 1/(2cal(A)^i) sum_(e in E(i)) l_e/d_e uvec_e^2 $,
  $ partial uvec^e = (2 u partial K)^e = rouge( ((partial K^i)/(cal(A)^i))_e med l_e/d_e uvec_e ) $,
  // centered flux divergence
  $ d_i = div (q uvec)_i = 1/cal(A)^i sum_(e in E(i)) n_(i e) (q_i)_e uvec^e $,
 $
   partial uvec_e = -(q grad partial d)_e =& rouge( -(q_i)_e delta_e (partial d^i)/cal(A)^i), \
  partial q^i = - (uvec dot grad partial d)^i =& bleu( -sum_(e in E(i)) uvec^e delta_e (partial d^i)/cal(A)^i )
 $,
)

= Pairings

== Continuous pairings

- scalar
$ braket(a,b) = integral_Sphere a b dSphere $
- vector
$ braket(uvec,vvec) = integral_Sphere uvec dot vvec dSphere $

== Discrete pairings

- scalar - density

$ 
  braket(a,b) 
  & simeq sum_i cal(A)^i med a_i b_i = sum_i underbracket(a^i, cal(A)^i a_i) b_i  \
  & simeq sum_v cal(A)_v med a_v b_v = sum_v underbracket(a^v, cal(A)_v a_v) b_v  \
  & simeq sum_e underbracket(cal(A)_e, 1/2 l_e d_e) med a_e b_e = sum_e underbracket(a^e, cal(A)_e a_e) b_e  \
$

- vector - covector

$
  (uvec dot vvec)_i simeq 1/(2A_i) sum_(e in E(i)) l_e d_e (uvec dot nvec_e )(vvec dot nvec_e)  
$
$
  braket(uvec,vvec) & simeq sum_i A_i (uvec dot vvec)_i \
  & = 1/2 sum_i sum_(e in E(i)) l_e d_e (uvec dot nvec_e )(vvec dot nvec_e) \
  & = sum_e rouge(underbracket(l_e (uvec dot nvec_e), uvec_e)) 
            bleu(underbracket(d_e (vvec dot nvec_e), v^e)) \

$
= Discrete adjoints

== Gradient
$
  grad : a_i & mapsto (grad a)_e = sum_i n_(e i) a_i \
          partial gvec^e & mapsto rouge((div partial gvec)^i =
          - sum_e n_(i e) partial gvec^e) \
          & sum_e partial gvec^e delta (grad a)_e = sum_ i rouge(sum_e partial gvec^e n_(e i)) delta a_i \
$
== Divergence
$
  div : uvec^e & mapsto (div u)_i = 1/cal(A)^i sum_e n_(i e) uvec^e \
          partial d^i & mapsto 
          rouge( (grad partial d)_e 
          = - sum_e n_(e i) (partial d^i) / cal(A)^i) \
          & sum_i partial d^i delta (div u)_i = sum_ e rouge(sum_i 1/cal(A)^i partial gvec^e n_(i e)) delta uvec^e \
$

== Curl

$ 
  uvec_e mapsto zeta^v &= sum_(e in E(v)) n_(v e)uvec_e \
  partial zeta_v mapsto rouge((-nabla^perp partial zeta)^e & = -sum_(v in V(e)) n_(e v) partial zeta_v ) \
  sum_v partial zeta_v delta zeta^v &= sum _v partial zeta_v sum_(e in E(v)) n_(v e)delta uvec_e 
  =  sum_e rouge(sum_(v in V(e)) partial zeta_v n_(v e))delta uvec_e \
$

== Squared covector

$
  uvec_e mapsto K_i & = 1/(2cal(A)^i) sum_(e in E(i)) l_e/d_e uvec_e^2 \
  partial K^i mapsto  & 
  rouge( 2u^e partial K_e 
  = l_e/d_e uvec_e sum_(i in C(e)) (partial K^i)/(cal(A)^i)) \

  sum_i partial K^i delta K_i 
  & = sum_i (partial K^i)/(cal(A)^i) sum_(e in E(i)) l_e/d_e uvec_e med delta uvec_e 
  = sum_e rouge( l_e/d_e uvec_e sum_(i in C(e)) (partial K^i)/(cal(A)^i))   med delta uvec_e 
$

== Average primal $arrow.r$ dual

$
  h_i mapsto h^v &= sum_(i in C(v)) cal(A)^(v i) h_i \
  partial h_v mapsto rouge( partial h^i &= sum_(v in V(i)) cal(A)^(i v) partial h_v) $
$
  sum_v partial h^v delta h_v &= sum_i rouge( sum_(v in V(i)) cal(A)^(i v) partial h_v) delta h_i
$

== Non-linear TRiSK
$
  (U^e, q_e) mapsto (q U^perp)_e 
    &= 1/2 sum_(e') w_(e e')(q_e+q_e')U^(e') \

  partial V^e mapsto 
  & rouge( (-q partial V^perp)_e = -1/2 sum_(e') w_(e e') (q_e+q_e') partial V^(e') ), \
  & bleu( (U times partial V)^e = 1/2 sum_(e') w_(e e')(partial V^e  U^(e') - U^e partial V^(e')))\ 
$
$
  sum_e partial V^e (q U^perp)_e 
  & = 1/2 sum_(e e') w_(e e') partial V^e  delta U^(e')(q_e+q_e') 
  + 1/2 sum_(e e') w_(e e') partial V^e  U^(e')(delta q_e+delta q_e') \
  & = sum_(e e') rouge(-1/2 w_(e e') partial V^(e') (q_e+q_e')) delta U^e 
  + sum_(e e') bleu( 1/2 w_(e e')(partial V^e  U^(e') - partial V^(e')  U^(e) )) delta q_e \
$

$
  integral partial vvec dot (delta q med kvec cprod uvec) 
  = 
  integral delta q med kvec  dot ( uvec cprod  partial vvec  ) 
$
$
  partial q = uvec cprod  partial vvec
$

== Divergence of centered flux

$
  (uvec^e, q_i) mapsto d_i = (div (q uvec))_i &= 1/cal(A)^i sum_(e in E(i)) n_(i e) uvec^e (q_i)_e \
  partial d^i mapsto 
      rouge( (-q grad partial d)_e &= -(q_i)_e sum_(i in C(e)) (partial d^i)/cal(A)^i n_(e i)), \
      bleu( - (uvec dot grad partial d)^i &= sum_(e in E(i)) ((partial d^i)/cal(A)^i - (partial d^j)/cal(A)_j) n_(i e) uvec^e ) \
$
$
  sum_i partial d^i delta d_i &= sum_(i,j) (partial d^i)/cal(A)^i sum_(e in E(i,j)) n_(i e) ( (q_i)_e delta uvec^e  + uvec^e (delta q_i+delta h_j)/2)\
  & = -  sum_e rouge( (q_i)_e sum_(i in C(e)) (partial d^i)/cal(A)^i n_(e i))  delta uvec^e \
  & + 1/2 sum_i bleu(sum_(e in E(i)) ((partial d^i)/cal(A)^i - (partial d^j)/cal(A)_j) n_(i e) uvec^e) delta q_i\
$

== Multiplied gradient

$
  (theta_i, pi_i) mapsto uvec_e = (theta grad pi)_e = (theta_i)_e nabla_e pi_i \
  partial uvec^e mapsto 
    partial theta^i = rouge( sum_(e in E(i)) partial uvec^e nabla_e pi_i ), spc 
    partial pi^i = bleu( -sum_(e in E(i)) (theta_i)_e n_(i e) partial uvec^e ), \
  \
  sum_e partial uvec^e delta uvec_e 
  = sum_e partial uvec^e 
  (theta_i)_e nabla_e delta pi_i  
  + sum_e partial uvec^e nabla_e pi_i (delta theta_i)_e \
  = sum_i rouge( sum_(e in E(i)) partial uvec^e nabla_e pi_i ) delta theta_i
    + sum_i bleu( -sum_(e in E(i)) (theta_i)_e n_(i e) partial uvec^e 
   ) delta pi_i
$