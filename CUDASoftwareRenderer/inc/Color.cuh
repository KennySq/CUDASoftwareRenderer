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

inline __device__ __host__ ColorRGBA ConvertDWORDToColor(const DWORD& texel)
{
	const float inverseMaxChannel = 1.0f / 255.999f;

	BYTE r = (texel >> 16);
	BYTE g = (texel >> 8);
	BYTE b = (texel >> 0);
	BYTE a = (texel >> 24);

	float x = r * inverseMaxChannel;
	float y = g * inverseMaxChannel;
	float z = b * inverseMaxChannel;
	float w = a * inverseMaxChannel;

	return ColorRGBA(x, y, z, w);
}

inline __device__ __host__ DWORD PackDepth(float depth)
{
	DWORD result = 0;

	BYTE b0 = reinterpret_cast<BYTE*>(&depth)[0];
	BYTE b1 = reinterpret_cast<BYTE*>(&depth)[1];
	BYTE b2 = reinterpret_cast<BYTE*>(&depth)[2];
	BYTE b3 = reinterpret_cast<BYTE*>(&depth)[3];

	result |= b0 << 0;
	result |= b1 << 8;
	result |= b2 << 16;
	result |= b3 << 24;

	return result;
}

inline __device__ __host__ float UnpackDepth(DWORD packed)
{
	return *reinterpret_cast<float*>(&packed);
}