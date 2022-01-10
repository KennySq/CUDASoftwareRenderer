#pragma once

#include"3DMath.cuh"
#include<FbxLoader.h>

struct SampleVertex
{
	__device__ __host__ SampleVertex()
	{

	}

	__device__ __host__ SampleVertex(const SampleVertex& other)
		: Position(other.Position), Normal(other.Normal), Texcoord(other.Texcoord)
	{

	}
	FLOAT3 Position;
	FLOAT3 Normal;
	FLOAT2 Texcoord;
};

struct VertexOutput
{
	__device__ __host__ VertexOutput()
	{

	}

	__device__ __host__ VertexOutput(FLOAT4 position, FLOAT4 normal, FLOAT2 texcoord)
		: Position(position), Normal(normal), Texcoord(texcoord)
	{

	}

	__device__ __host__ VertexOutput(const VertexOutput& other)
		: Position(other.Position), Normal(other.Normal), Texcoord(other.Texcoord)
	{

	}

	FLOAT4 Position;
	FLOAT4 Normal;
	FLOAT2 Texcoord;

};

struct AABB
{
	__device__ __host__ AABB()
	{

	}

	__device__ __host__ AABB(const INT2& _min, const INT2& _max)
		: Min(_min), Max(_max)
	{

	}

	__device__ __host__ AABB(const AABB& right)
		: Min(right.Min), Max(right.Max)
	{

	}
	INT2 Min;
	INT2 Max;
};

inline SampleVertex ConvertVertex(const Vertex& v)
{
	SampleVertex vertex;

	vertex.Position = FLOAT3(v.mPosition.x, v.mPosition.y, v.mPosition.z);
	vertex.Normal = FLOAT3(v.mNormal.x, v.mNormal.y, v.mNormal.z);
	vertex.Texcoord = FLOAT2(v.mTexcoord.x, v.mTexcoord.y);

	return vertex;
}