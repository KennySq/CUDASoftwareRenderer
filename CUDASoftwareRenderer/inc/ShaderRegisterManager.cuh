#pragma once
#include<thrust\device_vector.h>

enum eRegisterType
{
	REGISTER_TEXTURE,
};
__device__ struct ShaderRegisterManager
{
	__device__ struct Register
	{
		__device__ Register()
			: Resource(nullptr), Index(-1), Width(0), Height(0)
		{

		}
		__device__ Register(void* resource, unsigned int index, unsigned int width, unsigned int height)
			: Resource(resource), Index(index), Width(width), Height(height)
		{

		}
		void* Resource;
		unsigned int Width;
		unsigned int Height;
		unsigned int Index;

	};

	__device__ ShaderRegisterManager()
		: mSize(0)
	{ 
	}
	__device__ ~ShaderRegisterManager()
	{
		delete[] mRegisters;
		mSize = 0;
	}

	__device__ void Set(void* texture, unsigned int index, unsigned int width, unsigned int height, eRegisterType regType)
	{
		assert(texture != nullptr);
		
		switch (regType)
		{
		case REGISTER_TEXTURE:
		{
			mRegisters[index] = Register(texture, index, width, height);
		}
			break;
		}
	}

	__device__ Register Get(unsigned int index, eRegisterType regType)
	{
		switch (regType)
		{
		case REGISTER_TEXTURE:
		{
			return mRegisters[index];
		}
		break;
		}
	}

private:
	Register mRegisters[128];
	unsigned int mSize;
};