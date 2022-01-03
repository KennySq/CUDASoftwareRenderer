#pragma once

#include<immintrin.h>

using namespace std;

struct INT2
{
	static INT2 Min(const INT2& from, const INT2& to)
	{
		return from.x + from.y < to.x + to.y ? from : to;
	}

	inline float Length()
	{
		return sqrtf(x * x + y * y);
	}

	inline bool operator<(const INT2& right)
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
	__vectorcall FLOAT2(float _x, float _y)
		: x(_x), y(_y)
	{
	}
	__vectorcall FLOAT2()
		: x(.0f), y(.0f)
	{

	}

	union
	{
		float r[2];
		struct
		{
			float x;
			float y;
		};
	};
};

struct FLOAT3
{
	__vectorcall FLOAT3(float _x, float _y, float _z)
		: x(_x), y(_y), z(_z)
	{

	}
	__vectorcall FLOAT3()
		: x(.0f), y(.0f), z(.0f)
	{

	}
	union
	{
		float r[3];
		struct
		{
			float x;
			float y;
			float z;
		};
	};
};

struct FLOAT4
{
	__vectorcall FLOAT4(float _x, float _y, float _z, float _w)
		: x(_x), y(_y), z(_z), w(_w)
	{

	}
	__vectorcall FLOAT4()
		: x(.0f), y(.0f), z(.0f), w(.0f)
	{

	}
	union
	{
		float r[4];
		struct
		{
			float x;
			float y;
			float z;
			float w;
		};
	};
};

struct __FLOAT2X2PRV
{
	__FLOAT2X2PRV(float s0, float s1, float s2, float s3)
		: _11(s0), _12(s1), _21(s2), _22(s3)
	{

	}
	union
	{
		float r[4];
		struct
		{
			float _11, _12;
			float _21, _22;
		};
	};
};

struct FLOAT3X3
{
	static FLOAT3X3 Identity()
	{
		return FLOAT3X3(1, 0, 0, 0, 1, 0, 0, 0, 1);
	}

	__vectorcall FLOAT3X3()
	{
	}

	__vectorcall FLOAT3X3(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7, float s8)
		: _11(s0), _12(s1), _13(s2), _21(s3), _22(s4), _23(s5), _31(s6), _32(s7), _33(s8)
	{

	}

	union
	{
		__m128 r;
		struct
		{
			float _11; float _12; float _13;
			float _21; float _22; float _23;
			float _31; float _32; float _33;
		};
	};

};

struct VECTOR
{
	VECTOR(float _x, float _y, float _z, float _w)
		: x(_x), y(_y), z(_z), w(_w)
	{

	}

	VECTOR(const FLOAT2& v)
		: x(v.x), y(v.y)
	{

	}

	VECTOR(const FLOAT3& v)
		: x(v.x), y(v.y), z(v.z)
	{

	}

	VECTOR(const FLOAT4& v)
		: x(v.x), y(v.y), z(v.z), w(v.w)
	{

	}

	union
	{
		__m128 r;
		struct
		{
			float x;
			float y;
			float z;
			float w;
		};
	};
};

struct FLOAT4X4
{
	static FLOAT4X4 Identity()
	{
		return FLOAT4X4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	}

	__vectorcall FLOAT4X4()
	{

	}
	__vectorcall FLOAT4X4(float s0, float s1, float s3, float s4, float s5, float s6, float s7, float s8, float s9, float s10, float s11, float s12, float s13, float s14, float s15, float s16)
		: _11(s0), _12(s1), _13(s3), _14(s4), _21(s5), _22(s6), _23(s7), _24(s8), _31(s9), _32(s10), _33(s11), _34(s12), _41(s13), _42(s14), _43(s15), _44(s16)
	{

	}
	union
	{
		FLOAT4 r[4];
		struct
		{
			float _11; float _12; float _13; float _14;
			float _21; float _22; float _23; float _24;
			float _31; float _32; float _33; float _34;
			float _41; float _42; float _43; float _44;
		};
	};
};

//---------------------------------------------------------------
// FLOAT2

inline FLOAT2 __vectorcall operator+(const FLOAT2& v1, const FLOAT2& v2)
{
	return FLOAT2(v1.x + v2.x, v1.y + v2.y);
}

inline void __vectorcall operator+=(FLOAT2& v1, const FLOAT2& v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
}

inline FLOAT2 __vectorcall operator-(const FLOAT2& v1, const FLOAT2& v2)
{
	return FLOAT2(v1.x - v2.x, v1.y - v2.y);
}

inline void __vectorcall operator-=(FLOAT2& v1, const FLOAT2& v2)
{
	v1.x -= v2.x;
	v1.y -= v2.y;
}


inline FLOAT2 __vectorcall operator*(const FLOAT2& v1, float s)
{
	return FLOAT2(v1.x * s, v1.y * s);
}

inline void __vectorcall operator*=(FLOAT2& v, float s)
{
	v.x *= s;
	v.y *= s;
}

inline FLOAT2 __vectorcall operator/(const FLOAT2& v1, float s)
{
	return FLOAT2(v1.x / s, v1.y / s);
}

inline void __vectorcall operator/=(FLOAT2& v, float s)
{
	v.x /= s;
	v.y /= s;
}

//---------------------------------------------------------------
// FLOAT3

inline FLOAT3 __vectorcall operator+(const FLOAT3& v1, const FLOAT3& v2)
{
	return FLOAT3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline void __vectorcall operator+=(FLOAT3& v1, const FLOAT3& v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
	v1.z += v2.z;
}

inline FLOAT3 __vectorcall operator-(const FLOAT3& v1, const FLOAT3& v2)
{
	return FLOAT3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline void __vectorcall operator-=(FLOAT3& v1, const FLOAT3& v2)
{
	v1.x -= v2.x;
	v1.y -= v2.y;
	v1.z -= v2.z;
}

inline FLOAT3 __vectorcall operator*(const FLOAT3& v, float s)
{
	return FLOAT3(v.x * s, v.y * s, v.z * s);
}

inline void __vectorcall operator*=(FLOAT3& v, float s)
{
	v.x *= s;
	v.y *= s;
	v.z *= s;
}

inline FLOAT3 __vectorcall operator/(const FLOAT3& v1, float s)
{
	return FLOAT3(v1.x / s, v1.y / s, v1.z / s);
}

inline void __vectorcall operator/=(FLOAT3& v, float s)
{
	v.x /= s;
	v.y /= s;
	v.z /= s;
}

//---------------------------------------------------------------
// FLOAT4

inline FLOAT4 __vectorcall operator+(const FLOAT4& v1, const FLOAT4& v2)
{
	return FLOAT4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

inline void __vectorcall operator+=(FLOAT4& v1, const FLOAT4& v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
	v1.z += v2.z;
	v1.w += v2.w;
}

inline FLOAT4 __vectorcall operator-(const FLOAT4& v1, const FLOAT4& v2)
{
	return FLOAT4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}

inline void __vectorcall operator-=(FLOAT4& v1, const FLOAT4& v2)
{
	v1.x -= v2.x;
	v1.y -= v2.y;
	v1.z -= v2.z;
	v1.w -= v2.w;
}

inline FLOAT4 __vectorcall operator*(const FLOAT4& v, float s)
{
	return FLOAT4(v.x * s, v.y * s, v.z * s, v.w * s);
}

inline void __vectorcall operator*=(FLOAT4& v, float s)
{
	v.x *= s;
	v.y *= s;
	v.z *= s;
	v.w *= s;
}

inline FLOAT4 __vectorcall operator/(const FLOAT4& v1, float s)
{
	return FLOAT4(v1.x / s, v1.y / s, v1.z / s, v1.w / s);
}

inline void __vectorcall operator/=(FLOAT4& v, float s)
{
	v.x /= s;
	v.y /= s;
	v.z /= s;
	v.w /= s;
}

//---------------------------------------------------------------
// other methods

FLOAT2 inline __vectorcall Float2Abs(const FLOAT2& v)
{
	return FLOAT2(abs(v.x), abs(v.y));
}

FLOAT3 inline __vectorcall Float3Abs(const FLOAT3& v)
{
	return FLOAT3(abs(v.x), abs(v.y), abs(v.z));
}

FLOAT4 inline __vectorcall Float4Abs(const FLOAT4& v)
{
	
	return FLOAT4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

float inline __vectorcall Float2Length(const FLOAT2& v)
{
	
	return sqrt(v.x * v.x + v.y * v.y);
}

float inline __vectorcall Float3Length(const FLOAT3& v)
{
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float inline __vectorcall Float4Length(const FLOAT4& v)
{
	
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

float inline __vectorcall Float2Dot(const FLOAT2& v1, const FLOAT2& v2)
{
	return v1.x * v2.x + v1.y * v2.y;
}
float inline __vectorcall Float3Dot(const FLOAT3& v1, const FLOAT3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

float inline __vectorcall Float4Dot(const FLOAT4& v1, const FLOAT4& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

FLOAT3 inline __vectorcall Float3Cross(const FLOAT3& v1, const FLOAT3& v2)
{
	static FLOAT3 right = FLOAT3(1, 0, 0), up = FLOAT3(0, 1, 0), forward = FLOAT3(0, 0, 1);

	FLOAT3 s0 = right * (v1.y * v2.z - v1.z * v2.y);
	FLOAT3 s1 = up * (v1.x * v2.z - v1.z * v2.x);
	FLOAT3 s3 = forward * (v1.x * v2.y - v1.y * v2.x);

	return s0 + s1 + s3;
}

FLOAT2 inline __vectorcall Float2Cross(const FLOAT2& v1, const FLOAT2& v2)
{
	static FLOAT2 right = FLOAT2(1, 0);

	FLOAT2 s0 = right * (v1.y * v2.x - v1.x * v2.y);

	return FLOAT2(s0.x, s0.x);
}

// -----------------------------------------------------------------
// Matrix 4x4

float inline __vectorcall Float2x2Determinant(const __FLOAT2X2PRV& m)
{
	return m._11 * m._22 - m._12 * m._21;
}

float inline __vectorcall Float3x3Determinant(const FLOAT3X3& m)
{
	__FLOAT2X2PRV a = __FLOAT2X2PRV(m._22, m._23, m._32, m._33);
	__FLOAT2X2PRV b = __FLOAT2X2PRV(m._21, m._23, m._31, m._33);
	__FLOAT2X2PRV c = __FLOAT2X2PRV(m._21, m._22, m._31, m._32);

	float s0 = m._11 * Float2x2Determinant(a);
	float s1 = m._12 * Float2x2Determinant(b);
	float s2 = m._13 * Float2x2Determinant(c);

	return s0 - s1 + s2;
}

float inline __vectorcall Float4x4Determinant(const FLOAT4X4& m)
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

FLOAT4X4 inline __vectorcall Float4x4Multiply(const FLOAT4X4& m1, const FLOAT4X4& m2)
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

FLOAT4X4 inline __vectorcall Float4x4Translate(const FLOAT3& translate)
{
	return FLOAT4X4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, translate.x, translate.y, translate.z, 1);
}

FLOAT4X4 inline __vectorcall Float4x4Scale(const FLOAT3& scale)
{
	return FLOAT4X4(scale.x, 0, 0, 0, 0, scale.y, 0, 0, 0, 0, scale.z, 0, 0, 0, 0, 1);
}

FLOAT4X4 inline __vectorcall Float4x4RotationX(float theta)
{
	return FLOAT4X4(1, 0, 0, 0, 0, cos(theta), sin(theta), 0, 0, -sin(theta), cos(theta), 0, 0, 0, 0, 1);
}

FLOAT4X4 inline __vectorcall Float4x4RotationY(float theta)
{
	return FLOAT4X4(cos(theta), 0, -sin(theta), 0, 0, 1, 0, 0, sin(theta), 0, cos(theta), 0, 0, 0, 0, 1);
}

FLOAT4X4 inline __vectorcall Float4x4RotationZ(float theta)
{
	return FLOAT4X4(cos(theta), -sin(theta), 0, 0, sin(theta), cos(theta), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
}

inline __device__ __host__ DWORD deviceConvertRGB(float r, float g, float b, float a)
{
	BYTE comp0 = r * 255.999f;
	BYTE comp1 = g * 255.999f;
	BYTE comp2 = b * 255.999f;
	BYTE comp3 = a * 255.999f;

	DWORD color = 0;

	color |= (comp3 << 24); // alpha first.
	color |= (comp0 << 16); // r
	color |= (comp1 << 8);  // g
	color |= (comp2 << 0);  // b

	return color;
}