#ifndef SIMD_AVX_H
#define SIMD_AVX_H

#include <immintrin.h>
#include <simd.h>

/*
  implementation of SIMDs for Intel-CPUs with AVX support:
  https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 */

namespace ASC_HPC
{

  template<>
  class SIMD<mask64,2>
  {
    __m128i mask;
  public:
    SIMD (__m128i _mask) : mask(_mask) { };
    SIMD (__m128d _mask) : mask(_mm_castpd_si128(_mask)) { ; }
    SIMD (mask64 m0, mask64 m1) : mask{_mm_set_epi64x(m1, m0)} { } // Constructor for 2 mask values (makes SIMD<mask64,4> work)
    auto Val() const { return mask; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&mask)[i] != 0; }

    SIMD<mask64, 1> Lo() const { return SIMD<mask64,1>((*this)[0]); }
    SIMD<mask64, 1> Hi() const { return SIMD<mask64,1>((*this)[1]); }
  };


  template<>
  class SIMD<mask64,4>
  {
    __m256i mask;
  public:
    SIMD (__m256i _mask) : mask(_mask) { };
    SIMD (__m256d _mask) : mask(_mm256_castpd_si256(_mask)) { ; }
    auto Val() const { return mask; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&mask)[i] != 0; }

    SIMD<mask64, 2> Lo() const { return SIMD<mask64,2>((*this)[0], (*this)[1]); }
    SIMD<mask64, 2> Hi() const { return SIMD<mask64,2>((*this)[2], (*this)[3]); }
  };




  template<>
  class SIMD<double,2>
  {
    __m128d val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(double _val) : val{_mm_set1_pd(_val)} {};
    SIMD(__m128d _val) : val{_val} {};
    SIMD (double v0, double v1) : val{_mm_set_pd(v1,v0)} {  }
    SIMD (SIMD<double,1> v0, SIMD<double,1> v1) : SIMD(v0[0], v0[1]) { }  // better with _mm256_set_m128d
    SIMD (std::array<double,2> a) : SIMD(a[0],a[1]) { }
    SIMD (double const * p) { val = _mm_loadu_pd(p); }
    SIMD (double const * p, SIMD<mask64,2> mask) { val = _mm_maskload_pd(p, mask.Val()); }
    
    static constexpr int Size() { return 2; }
    auto Val() const { return val; }
    const double * Ptr() const { return (double*)&val; }
    SIMD<double, 1> Lo() const { return SIMD<double,1>((*this)[0]); }
    SIMD<double, 1> Hi() const { return SIMD<double,1>((*this)[1]); }

    // better:
    // SIMD<double, 2> Lo() const { return _mm256_extractf128_pd(val, 0); }
    // SIMD<double, 2> Hi() const { return _mm256_extractf128_pd(val, 1); }
    double operator[](size_t i) const { return ((double*)&val)[i]; }

    void Store (double * p) const { _mm_storeu_pd(p, val); }
    void Store (double * p, SIMD<mask64,2> mask) const { _mm_maskstore_pd(p, mask.Val(), val); }
  };

  
  template<>
  class SIMD<double,4>
  {
    __m256d val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(double _val) : val{_mm256_set1_pd(_val)} {};
    SIMD(__m256d _val) : val{_val} {};
    SIMD (double v0, double v1, double v2, double v3) : val{_mm256_set_pd(v3,v2,v1,v0)} {  }
    SIMD (SIMD<double,2> v0, SIMD<double,2> v1) : SIMD(v0[0], v0[1], v1[0], v1[1]) { }  // better with _mm256_set_m128d
    SIMD (std::array<double,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    SIMD (double const * p) { val = _mm256_loadu_pd(p); }
    SIMD (double const * p, SIMD<mask64,4> mask) { val = _mm256_maskload_pd(p, mask.Val()); }
    
    static constexpr int Size() { return 4; }
    auto Val() const { return val; }
    const double * Ptr() const { return (double*)&val; }
    SIMD<double, 2> Lo() const { return SIMD<double,2>((*this)[0], (*this)[1]); }
    SIMD<double, 2> Hi() const { return SIMD<double,2>((*this)[2], (*this)[3]); }

    // better:
    // SIMD<double, 2> Lo() const { return _mm256_extractf128_pd(val, 0); }
    // SIMD<double, 2> Hi() const { return _mm256_extractf128_pd(val, 1); }
    double operator[](size_t i) const { return ((double*)&val)[i]; }

    void Store (double * p) const { _mm256_storeu_pd(p, val); }
    void Store (double * p, SIMD<mask64,4> mask) const { _mm256_maskstore_pd(p, mask.Val(), val); }

    // TODO: Transpose
    /* void Transpose (SIMD<double,4> a0, SIMD<double,4> a1, SIMD<double,4> a2, SIMD<double,4> a3,
                SIMD<double,4> &b0, SIMD<double,4> &b1, SIMD<double,4> &b2, SIMD<double,4> &b3);*/
    
  };
  



  template<>
  class SIMD<int64_t,2>
  {
    __m128i val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(int64_t _val) : val{_mm_set1_epi64x(_val)} {};
    SIMD(__m128i _val) : val{_val} {};
    SIMD (int64_t v0, int64_t v1) : val{_mm_set_epi64x(v1,v0) } { } 
    SIMD (SIMD<int64_t,1> v0, SIMD<int64_t,1> v1) : SIMD(v0[0], v1[0]) { }  // can do better !
    // SIMD (std::array<double,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    // SIMD (double const * p) { val = _mm256_loadu_pd(p); }
    // SIMD (double const * p, SIMD<mask64,4> mask) { val = _mm256_maskload_pd(p, mask.Val()); }
    
    static constexpr int Size() { return 4; }
    auto Val() const { return val; }
    // const double * Ptr() const { return (double*)&val; }
    // SIMD<double, 2> Lo() const { return _mm256_extractf128_pd(val, 0); }
    // SIMD<double, 2> Hi() const { return _mm256_extractf128_pd(val, 1); }
    int64_t operator[](size_t i) const { return ((int64_t*)&val)[i]; }
  };

  
  template<>
  class SIMD<int64_t,4>
  {
    __m256i val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(int64_t _val) : val{_mm256_set1_epi64x(_val)} {};
    SIMD(__m256i _val) : val{_val} {};
    SIMD (int64_t v0, int64_t v1, int64_t v2, int64_t v3) : val{_mm256_set_epi64x(v3,v2,v1,v0) } { } 
    SIMD (SIMD<int64_t,2> v0, SIMD<int64_t,2> v1) : SIMD(v0[0], v0[1], v1[0], v1[1]) { }  // can do better !
    // SIMD (std::array<double,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    // SIMD (double const * p) { val = _mm256_loadu_pd(p); }
    // SIMD (double const * p, SIMD<mask64,4> mask) { val = _mm256_maskload_pd(p, mask.Val()); }
    
    static constexpr int Size() { return 4; }
    auto Val() const { return val; }
    // const double * Ptr() const { return (double*)&val; }
    // SIMD<double, 2> Lo() const { return _mm256_extractf128_pd(val, 0); }
    // SIMD<double, 2> Hi() const { return _mm256_extractf128_pd(val, 1); }
    int64_t operator[](size_t i) const { return ((int64_t*)&val)[i]; }
  };
  


  template <int64_t first>
  class IndexSequence<int64_t, 2, first> : public SIMD<int64_t,2>
  {
  public:
    IndexSequence()
      : SIMD<int64_t,2> (first, first+1) { }
  };

  template <int64_t first>
  class IndexSequence<int64_t, 4, first> : public SIMD<int64_t,4>
  {
  public:
    IndexSequence()
      : SIMD<int64_t,4> (first, first+1, first+2, first+3) { }
  };
  // Should we do this for 2-Vector (_m128) too?


  inline auto operator+ (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_add_pd(a.Val(), b.Val())); }
  inline auto operator- (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_sub_pd(a.Val(), b.Val())); }
  inline auto operator/ (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_div_pd(a.Val(), b.Val())); }
  inline auto operator* (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_mul_pd(a.Val(), b.Val())); }
  inline auto operator* (double a, SIMD<double,2> b) { return SIMD<double,2>(a)*b; }
  inline auto operator/ (double a, SIMD<double,2> b) { return SIMD<double,2>(a)/b; }

  
  inline auto operator+ (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_add_pd(a.Val(), b.Val())); }
  inline auto operator- (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_sub_pd(a.Val(), b.Val())); }
  inline auto operator/ (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_div_pd(a.Val(), b.Val())); }
  inline auto operator* (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_mul_pd(a.Val(), b.Val())); }
  inline auto operator* (double a, SIMD<double,4> b) { return SIMD<double,4>(a)*b; }
  inline auto operator/ (double a, SIMD<double,4> b) { return SIMD<double,4>(a)/b; }


// do we need 2-vector version of this?
#ifdef __FMA__
  inline SIMD<double,4> FMA (SIMD<double,4> a, SIMD<double,4> b, SIMD<double,4> c)
  { return _mm256_fmadd_pd (a.Val(), b.Val(), c.Val()); }
#endif


  inline SIMD<mask64,2> operator>= (SIMD<int64_t,2> a , SIMD<int64_t,2> b)
  { // there is no a>=b, so we return !(b>a)
    return  _mm_xor_si128(_mm_cmpgt_epi64(b.Val(),a.Val()),_mm_set1_epi32(-1)); }
  
  inline auto operator>= (SIMD<double,2> a, SIMD<double,2> b)
  { return SIMD<mask64,2>(_mm_cmp_pd (a.Val(), b.Val(), _CMP_GE_OQ)); }

   inline SIMD<mask64,2> operator> (SIMD<int64_t,2> a , SIMD<int64_t,2> b)
  { 
    return  _mm_cmpgt_epi64(a.Val(),b.Val()); }
  
  inline auto operator> (SIMD<double,2> a, SIMD<double,2> b)
  { return SIMD<mask64,2>(_mm_cmp_pd (a.Val(), b.Val(), _CMP_GT_OQ)); }



  inline SIMD<mask64,4> operator>= (SIMD<int64_t,4> a , SIMD<int64_t,4> b)
  { // there is no a>=b, so we return !(b>a)
    return  _mm256_xor_si256(_mm256_cmpgt_epi64(b.Val(),a.Val()),_mm256_set1_epi32(-1)); }
  
  inline auto operator>= (SIMD<double,4> a, SIMD<double,4> b)
  { return SIMD<mask64,4>(_mm256_cmp_pd (a.Val(), b.Val(), _CMP_GE_OQ)); }

   inline SIMD<mask64,4> operator> (SIMD<int64_t,4> a , SIMD<int64_t,4> b)
  { 
    return  _mm256_cmpgt_epi64(a.Val(),b.Val()); }
  
  inline auto operator> (SIMD<double,4> a, SIMD<double,4> b)
  { return SIMD<mask64,4>(_mm256_cmp_pd (a.Val(), b.Val(), _CMP_GT_OQ)); }
  
  
}

#endif
