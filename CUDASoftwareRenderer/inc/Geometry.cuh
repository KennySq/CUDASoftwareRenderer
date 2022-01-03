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

inline SampleVertex ConvertVertex(const Vertex& v)
{
	SampleVertex vertex;

	vertex.Position = FLOAT3(v.mPosition.x, v.mPosition.y, v.mPosition.z);
	vertex.Normal = FLOAT3(v.mNormal.x, v.mNormal.y, v.mNormal.z);
	vertex.Texcoord = FLOAT2(v.mTexcoord.x, v.mTexcoord.y);

	return vertex;
}