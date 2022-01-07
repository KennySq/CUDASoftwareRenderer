#pragma once

#include<immintrin.h>

using namespace std;

struct INT2
{
	static  __device__ __host__ INT2 Min(const INT2& from, const INT2& to)
	{
		return from.x + from.y < to.x + to.y ? from : to;
	}

	static  __device__ __host__ INT2 Max(const INT2& from, const INT2& to)
	{
		return from.x + from.y > to.x + to.y ? from : to;
	}

	__device__ __host__ inline float Length()
	{
		return sqrtf(x * x + y * y);
	}

	__device__ __host__ inline bool operator<(const INT2& right)
	{
		return x + y < right.x + right.y;
	}
	__device__ __host__ INT2(int _x, int _y)
		: x(_x), y(_y)
	{

	}

	__device__ __host__ INT2(const INT2& other)
		: x(other.x), y(other.y)
	{

	}

	int x;
	int y;
};

struct FLOAT2
{
	__device__ __host__ FLOAT2(const FLOAT2& right)
		: x(right.x), y(right.y)
	{

	}

	__device__ __host__  FLOAT2(float _x, float _y)
		: x(_x), y(_y)
	{
	}
	__device__ __host__  FLOAT2()
		: x(.0f), y(.0f)
	{

	}

	float x;
	float y;
};

struct FLOAT3
{
	__device__ __host__  FLOAT3(float _x, float _y, float _z)
		: x(_x), y(_y), z(_z)
	{

	}
	__device__ __host__  FLOAT3()
		: x(.0f), y(.0f), z(.0f)
	{

	}

	float x;
	float y;
	float z;
};

struct FLOAT4
{
	__device__ __host__ FLOAT4(const FLOAT4& right)
		: x(right.x), y(right.y), z(right.z), w(right.w)
	{

	}

	__device__ __host__  FLOAT4(float _x, float _y, float _z, float _w)
		: x(_x), y(_y), z(_z), w(_w)
	{

	}
	__device__ __host__  FLOAT4()
		: x(.0f), y(.0f), z(.0f), w(.0f)
	{

	}

	float x;
	float y;
	float z;
	float w;

};

struct __FLOAT2X2PRV
{
	__device__ __host__ __FLOAT2X2PRV(float s0, float s1, float s2, float s3)
		: _11(s0), _12(s1), _21(s2), _22(s3)
	{

	}

	float _11, _12;
	float _21, _22;

};

struct FLOAT3X3
{
	static FLOAT3X3 Identity()
	{
		return FLOAT3X3(1, 0, 0, 0, 1, 0, 0, 0, 1);
	}

	__device__ __host__  FLOAT3X3()
	{
	}

	__device__ __host__  FLOAT3X3(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7, float s8)
		: _11(s0), _12(s1), _13(s2), _21(s3), _22(s4), _23(s5), _31(s6), _32(s7), _33(s8)
	{

	}

	float _11; float _12; float _13;
	float _21; float _22; float _23;
	float _31; float _32; float _33;

};

struct VECTOR
{
	__device__ __host__ VECTOR(float _x, float _y, float _z, float _w)
		: x(_x), y(_y), z(_z), w(_w)
	{

	}

	__device__ __host__ VECTOR(const FLOAT2& v)
		: x(v.x), y(v.y)
	{

	}

	__device__ __host__ VECTOR(const FLOAT3& v)
		: x(v.x), y(v.y), z(v.z)
	{

	}

	__device__ __host__ VECTOR(const FLOAT4& v)
		: x(v.x), y(v.y), z(v.z), w(v.w)
	{

	}

	float x;
	float y;
	float z;
	float w;
};

struct FLOAT4X4
{
	static __device__ __host__ FLOAT4X4 Identity()
	{
		return FLOAT4X4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	}

	__device__ __host__  FLOAT4X4()
	{

	}
	__device__ __host__  FLOAT4X4(float s0, float s1, float s3, float s4, float s5, float s6, float s7, float s8, float s9, float s10, float s11, float s12, float s13, float s14, float s15, float s16)
		: _11(s0), _12(s1), _13(s3), _14(s4), _21(s5), _22(s6), _23(s7), _24(s8), _31(s9), _32(s10), _33(s11), _34(s12), _41(s13), _42(s14), _43(s15), _44(s16)
	{

	}
	float _11; float _12; float _13; float _14;
	float _21; float _22; float _23; float _24;
	float _31; float _32; float _33; float _34;
	float _41; float _42; float _43; float _44;
};

//---------------------------------------------------------------
// FLOAT2

inline __device__ __host__ FLOAT2  operator+(const FLOAT2& v1, const FLOAT2& v2)
{
	return FLOAT2(v1.x + v2.x, v1.y + v2.y);
}

inline __device__ __host__ void  operator+=(FLOAT2& v1, const FLOAT2& v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
}

inline __device__ __host__ FLOAT2  operator-(const FLOAT2& v1, const FLOAT2& v2)
{
	return FLOAT2(v1.x - v2.x, v1.y - v2.y);
}

inline __device__ __host__ void  operator-=(FLOAT2& v1, const FLOAT2& v2)
{
	v1.x -= v2.x;
	v1.y -= v2.y;
}


inline __device__ __host__ FLOAT2  operator*(const FLOAT2& v1, float s)
{
	return FLOAT2(v1.x * s, v1.y * s);
}

inline __device__ __host__ void  operator*=(FLOAT2& v, float s)
{
	v.x *= s;
	v.y *= s;
}

inline __device__ __host__ FLOAT2  operator/(const FLOAT2& v1, float s)
{
	return FLOAT2(v1.x / s, v1.y / s);
}

inline __device__ __host__ void  operator/=(FLOAT2& v, float s)
{
	v.x /= s;
	v.y /= s;
}

//---------------------------------------------------------------
// FLOAT3

inline __device__ __host__ FLOAT3  operator+(const FLOAT3& v1, const FLOAT3& v2)
{
	return FLOAT3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __device__ __host__ void  operator+=(FLOAT3& v1, const FLOAT3& v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
	v1.z += v2.z;
}

inline __device__ __host__ FLOAT3  operator-(const FLOAT3& v1, const FLOAT3& v2)
{
	return FLOAT3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline __device__ __host__ void  operator-=(FLOAT3& v1, const FLOAT3& v2)
{
	v1.x -= v2.x;
	v1.y -= v2.y;
	v1.z -= v2.z;
}

inline __device__ __host__ FLOAT3  operator*(const FLOAT3& v, float s)
{
	return FLOAT3(v.x * s, v.y * s, v.z * s);
}

inline __device__ __host__ void  operator*=(FLOAT3& v, float s)
{
	v.x *= s;
	v.y *= s;
	v.z *= s;
}

inline __device__ __host__ FLOAT3  operator/(const FLOAT3& v1, float s)
{
	return FLOAT3(v1.x / s, v1.y / s, v1.z / s);
}

inline __device__ __host__ void  operator/=(FLOAT3& v, float s)
{
	v.x /= s;
	v.y /= s;
	v.z /= s;
}

//---------------------------------------------------------------
// FLOAT4

inline __device__ __host__ FLOAT4  operator+(const FLOAT4& v1, const FLOAT4& v2)
{
	return FLOAT4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

inline __device__ __host__ void  operator+=(FLOAT4& v1, const FLOAT4& v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
	v1.z += v2.z;
	v1.w += v2.w;
}

inline __device__ __host__ FLOAT4  operator-(const FLOAT4& v1, const FLOAT4& v2)
{
	return FLOAT4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}

inline __device__ __host__ void  operator-=(FLOAT4& v1, const FLOAT4& v2)
{
	v1.x -= v2.x;
	v1.y -= v2.y;
	v1.z -= v2.z;
	v1.w -= v2.w;
}

inline __device__ __host__ FLOAT4  operator*(const FLOAT4& v, float s)
{
	return FLOAT4(v.x * s, v.y * s, v.z * s, v.w * s);
}

inline __device__ __host__ void  operator*=(FLOAT4& v, float s)
{
	v.x *= s;
	v.y *= s;
	v.z *= s;
	v.w *= s;
}

inline __device__ __host__ FLOAT4  operator/(const FLOAT4& v1, float s)
{
	return FLOAT4(v1.x / s, v1.y / s, v1.z / s, v1.w / s);
}

inline __device__ __host__ void  operator/=(FLOAT4& v, float s)
{
	v.x /= s;
	v.y /= s;
	v.z /= s;
	v.w /= s;
}

//---------------------------------------------------------------
// other methods

FLOAT2 __device__ __host__ inline  Float2Abs(const FLOAT2& v)
{
	return FLOAT2(abs(v.x), abs(v.y));
}

FLOAT3 __device__ __host__ inline  Float3Abs(const FLOAT3& v)
{
	return FLOAT3(abs(v.x), abs(v.y), abs(v.z));
}

FLOAT4 __device__ __host__ inline  Float4Abs(const FLOAT4& v)
{

	return FLOAT4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

float __device__ __host__ inline  Float2Length(const FLOAT2& v)
{

	return sqrt(v.x * v.x + v.y * v.y);
}

float __device__ __host__ inline  Float3Length(const FLOAT3& v)
{
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float __device__ __host__ inline  Float4Length(const FLOAT4& v)
{

	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

float __device__ __host__ inline  Float2Dot(const FLOAT2& v1, const FLOAT2& v2)
{
	return v1.x * v2.x + v1.y * v2.y;
}
float __device__ __host__ inline  Float3Dot(const FLOAT3& v1, const FLOAT3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

float __device__ __host__ inline  Float4Dot(const FLOAT4& v1, const FLOAT4& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

FLOAT4 __device__ __host__ inline  Float4Multiply(const FLOAT4& v, const FLOAT4X4& m)
{
	FLOAT4 result;

	result.x = v.x * m._11 + v.y * m._21 + v.z * m._31 + v.w * m._41;
	result.y = v.x * m._12 + v.y * m._22 + v.z * m._32 + v.w * m._42;
	result.z = v.x * m._13 + v.y * m._23 + v.z * m._33 + v.w * m._43;
	result.w = v.x * m._14 + v.y * m._24 + v.z * m._34 + v.w * m._44;

	return result;
}

FLOAT3 __device__ __host__ inline  Float3Cross(const FLOAT3& v1, const FLOAT3& v2)
{
	FLOAT3 right = FLOAT3(1, 0, 0), up = FLOAT3(0, 1, 0), forward = FLOAT3(0, 0, 1);

	FLOAT3 s0 = right * (v1.y * v2.z - v1.z * v2.y);
	FLOAT3 s1 = up * (v1.x * v2.z - v1.z * v2.x);
	FLOAT3 s3 = forward * (v1.x * v2.y - v1.y * v2.x);

	return s0 + s1 + s3;
}

FLOAT2 __device__ __host__ inline  Float2Cross(const FLOAT2& v1, const FLOAT2& v2)
{
	FLOAT2 right = FLOAT2(1, 0);

	FLOAT2 s0 = right * (v1.y * v2.x - v1.x * v2.y);

	return FLOAT2(s0.x, s0.x);
}

// -----------------------------------------------------------------
// Matrix 4x4

float __device__ __host__ inline  Float2x2Determinant(const __FLOAT2X2PRV& m)
{
	return m._11 * m._22 - m._12 * m._21;
}

float __device__ __host__ inline  Float3x3Determinant(const FLOAT3X3& m)
{
	__FLOAT2X2PRV a = __FLOAT2X2PRV(m._22, m._23, m._32, m._33);
	__FLOAT2X2PRV b = __FLOAT2X2PRV(m._21, m._23, m._31, m._33);
	__FLOAT2X2PRV c = __FLOAT2X2PRV(m._21, m._22, m._31, m._32);

	float s0 = m._11 * Float2x2Determinant(a);
	float s1 = m._12 * Float2x2Determinant(b);
	float s2 = m._13 * Float2x2Determinant(c);

	return s0 - s1 + s2;
}

float __device__ __host__ inline  Float4x4Determinant(const FLOAT4X4& m)
{
	FLOAT3X3 a = FLOAT3X3(m._22, m._23, m._24, m._32, m._33, m._34, m._42, m._43, m._44);
	FLOAT3X3 b = FLOAT3X3(m._12, m._13, m._14, m._32, m._33, m._34, m._42, m._43, m._44);
	FLOAT3X3 c = FLOAT3X3(m._12, m._13, m._14, m._22, m._23, m._24, m._42, m._43, m._44);
	FLOAT3X3 d = FLOAT3X3(m._12, m._13, m._14, m._22, m._23, m._24, m._32, m._33, m._34);

	float s0 = m._11 * Float3x3Determinant(a);
	float s1 = m._21 * Float3x3Determinant(b);
	float s2 = m._31 * Float3x3Determinant(c);
	float s3 = m._41 * Float3x3Determinant(d);

	return s0 - s1 + s2 - s3;
}

FLOAT4X4 __device__ __host__ inline  Float4x4Multiply(const FLOAT4X4& m1, const FLOAT4X4& m2)
{
	FLOAT4X4 mat;

	mat._11 = m1._11 * m2._11 + m1._12 * m2._21 + m1._13 * m2._31 + m1._14 * m2._41;
	mat._12 = m1._11 * m2._12 + m1._12 * m2._22 + m1._13 * m2._32 + m1._14 * m2._42;
	mat._13 = m1._11 * m2._13 + m1._12 * m2._23 + m1._13 * m2._33 + m1._14 * m2._43;
	mat._14 = m1._11 * m2._14 + m1._12 * m2._24 + m1._13 * m2._34 + m1._14 * m2._44;

	mat._21 = m1._21 * m2._11 + m1._22 * m2._21 + m1._23 * m2._31 + m1._24 * m2._41;
	mat._22 = m1._21 * m2._12 + m1._22 * m2._22 + m1._23 * m2._32 + m1._24 * m2._42;
	mat._23 = m1._21 * m2._13 + m1._22 * m2._23 + m1._23 * m2._33 + m1._24 * m2._43;
	mat._24 = m1._21 * m2._14 + m1._22 * m2._24 + m1._23 * m2._34 + m1._24 * m2._44;

	mat._31 = m1._31 * m2._11 + m1._32 * m2._21 + m1._33 * m2._31 + m1._34 * m2._41;
	mat._32 = m1._31 * m2._12 + m1._32 * m2._22 + m1._33 * m2._32 + m1._34 * m2._42;
	mat._33 = m1._31 * m2._13 + m1._32 * m2._23 + m1._33 * m2._33 + m1._34 * m2._43;
	mat._34 = m1._31 * m2._14 + m1._32 * m2._24 + m1._33 * m2._34 + m1._34 * m2._44;

	mat._41 = m1._41 * m2._11 + m1._42 * m2._21 + m1._43 * m2._31 + m1._44 * m2._41;
	mat._42 = m1._41 * m2._12 + m1._42 * m2._22 + m1._43 * m2._32 + m1._44 * m2._42;
	mat._43 = m1._41 * m2._13 + m1._42 * m2._23 + m1._43 * m2._33 + m1._44 * m2._43;
	mat._44 = m1._41 * m2._14 + m1._42 * m2._24 + m1._43 * m2._34 + m1._44 * m2._44;

	return mat;
}

FLOAT4X4 __device__ __host__ inline  Float4x4ViewMatrix(float pitch, float yaw, float roll)
{
	FLOAT4X4 yawMat(cos(yaw), 0, sin(yaw), 0, 0, 1, 0, 0, -sin(yaw), 0, cos(yaw), 0, 0, 0, 0, 1);
	FLOAT4X4 pitchMat(1, 0, 0, 0, 0, cos(pitch), -sin(pitch), 0, 0, sin(pitch), cos(pitch), 0, 0, 0, 0, 1);
	FLOAT4X4 rollMat(cos(roll), -sin(roll), 0, 0, sin(roll), cos(roll), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

	FLOAT4X4 result = Float4x4Multiply(yawMat, pitchMat);

	return Float4x4Multiply(result, rollMat);
}

FLOAT4X4 __device__ __host__ inline  Float4x4ProjectionMatrix(float n, float f, float fov, float aspectRatio)
{
	float focalLength = 1.0f / tan(fov * 0.5f);

	return FLOAT4X4(focalLength / aspectRatio, 0, 0, 0,
		0, focalLength, 0, 0,
		0, 0, ((n + f) / (n - f)), ((2.0f * n * f) / (n - f)),
		0, 0, -1, 0);
}

FLOAT4X4 __device__ __host__ inline  Float4x4Transpose(const FLOAT4X4& other)
{
	return FLOAT4X4(other._11, other._21, other._31, other._41,
		other._12, other._22, other._32, other._42,
		other._13, other._23, other._33, other._43,
		other._14, other._24, other._34, other._44);
}

float __device__ __host__ inline DegreeToRadian(float deg)
{
	return deg * 3.141592f / 180.0f;
}


FLOAT4X4 __device__ __host__ inline  Float4x4Translate(const FLOAT3& translate)
{
	return FLOAT4X4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, translate.x, translate.y, translate.z, 1);
}

FLOAT4X4 __device__ __host__ inline  Float4x4Scale(const FLOAT3& scale)
{
	return FLOAT4X4(scale.x, 0, 0, 0, 0, scale.y, 0, 0, 0, 0, scale.z, 0, 0, 0, 0, 1);
}

FLOAT4X4 __device__ __host__ inline  Float4x4RotationX(float theta)
{
	return FLOAT4X4(1, 0, 0, 0, 0, cos(theta), sin(theta), 0, 0, -sin(theta), cos(theta), 0, 0, 0, 0, 1);
}

FLOAT4X4 __device__ __host__ inline  Float4x4RotationY(float theta)
{
	return FLOAT4X4(cos(theta), 0, -sin(theta), 0, 0, 1, 0, 0, sin(theta), 0, cos(theta), 0, 0, 0, 0, 1);
}

FLOAT4X4 __device__ __host__ inline  Float4x4RotationZ(float theta)
{
	return FLOAT4X4(cos(theta), -sin(theta), 0, 0, sin(theta), cos(theta), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
}

inline __device__ __host__ DWORD deviceConvertRGB(float r, float g, float b, float a)
{
	BYTE comp0 = static_cast<BYTE>(r * 255.999f);
	BYTE comp1 = static_cast<BYTE>(g * 255.999f);
	BYTE comp2 = static_cast<BYTE>(b * 255.999f);
	BYTE comp3 = static_cast<BYTE>(a * 255.999f);

	DWORD color = 0;

	color |= (comp3 << 24); // alpha first.
	color |= (comp0 << 16); // r
	color |= (comp1 << 8);  // g
	color |= (comp2 << 0);  // b

	return color;
}

template<typename _Ty>
__device__ __host__ inline void Clamp(_Ty& t, _Ty min, _Ty max)
{
	if (t < min)
	{
		t = min;
	}
	if (t > max)
	{
		t = max;
	}
}