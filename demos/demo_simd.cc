#include <iostream>
#include <sstream>


#include <simd.h>
#include <simd_avx.h>

using namespace ASC_HPC;
using std::cout, std::endl;

auto Func1 (SIMD<double> a, SIMD<double> b)
{
  return a+b;
}

auto Func2 (SIMD<double,4> a, SIMD<double,4> b)
{
  return a+3*b;
}

auto Func3 (SIMD<double,4> a, SIMD<double,4> b, SIMD<double,4> c)
{
  return FMA(a,b,c);
}


auto Load (double * p)
{
  return SIMD<double,2>(p);
}

auto LoadMask (double * p, SIMD<mask64,2> m)
{
  return SIMD<double,2>(p, m);
}

auto TestSelect (SIMD<mask64,2> m, SIMD<double,2> a, SIMD<double,2> b)
{
  return Select (m, a, b);
}

SIMD<double,2> TestHSum (SIMD<double,4> a, SIMD<double,4> b)
{
  return HSum(a,b);
}



int main()
{
  SIMD<double,4> a(1.,2.,3.,4.);
  SIMD<double,4> b(1.0);
  
  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
  cout << "a+b = " << a+b << endl;
/*
  cout << "HSum(a) = " << HSum(a) << endl;
  cout << "HSum(a,b) = " << HSum(a,b) << endl;

  double mem[4] = { 10, 11, 12, 13 };
  SIMD<double,4> c(&mem[0]);
  auto d = a*b+c;
  auto e = a*b-c;
  auto f = a/c;
  std::cout << "a*b +c = " << d << std::endl;
  std::cout << "a*b -c = " << e << std::endl;
  std::cout << "a/b = " << f << std::endl;

  auto sequ = IndexSequence<int64_t, 4>();
  cout << "sequ = " << sequ << endl;
  auto mask = (2 >= sequ);
  auto mask2 = (2 > sequ);
  cout << "2 >= " << sequ << " = " << mask << endl;
  cout << "2 > " << sequ << " = " << mask2 << endl;

  {
    double a[] = { 10, 10, 10, 10 };
    SIMD<double,4> sa(&a[0], mask);
    cout << "sa = " << sa << endl;
  }

  cout << "Select(mask, a, b) = " << Select(mask, a,b) << endl;

// TODO: missing arithmetic functions?
  cout << "a*b/c = " << a*b/c << endl;
  double res[4];
  d.Store(&res[0]);
  cout << d << endl;*/
  //TO DO: bei simd_avx.h operator> definieren!!

// SIMD<T,2> Klasse
  SIMD<double,2> e(2.,5.);
  SIMD<double,2> f(1.0);
  
  cout << "e = " << e << endl;
  cout << "f = " << f << endl;
  cout << "e+f = " << e+f << endl;
  cout << "f/e = " << f/e << endl;
  cout << "3/e = " << 3.0/e << endl;

  auto sequ_new = IndexSequence<int64_t, 2>();
  cout << "sequ = " << sequ_new << endl;
  auto mask_new = (1 >= sequ_new);
  auto mask2_new = (1 > sequ_new);
  cout << "2 >= " << sequ_new << " = " << mask_new << endl;
  cout << "2 > " << sequ_new << " = " << mask2_new << endl;
}
