#pragma once

struct ColorRGBA
{
	__device__ __host__ ColorRGBA(float _r, float _g, float _b, float _a)
		: r(_r), g(_g), b(_b), a(_a)
	{

	}

	__device__ __host__ ColorRGBA(const ColorRGBA& right)
		: r(right.r), g(right.g), b(right.b), a(right.a)
	{

	}

	float r, g, b, a;

	bool operator==(const ColorRGBA& right)
	{
		return r == right.r && g == right.g && b == right.b && a == right.a;
	}
};

inline __device__ __host__ DWORD ConvertColorToDWORD(const ColorRGBA& color)
{
	BYTE r = color.r * 255.999f;
	BYTE g = color.g * 255.999f;
	BYTE b = color.b * 255.999f;
	BYTE a = color.a * 255.999f;

	DWORD result = 0;

	result |= a << 24;
	result |= r << 16;
	result |= g << 8;
	result |= b << 0;

	return result;
}