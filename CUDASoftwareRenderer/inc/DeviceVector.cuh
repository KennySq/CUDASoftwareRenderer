#pragma once

template<typename _Ty>
class DeviceVector
{
public:
	__device__ DeviceVector(size_t capacity)
		: mSize(sizeof(_Ty)* capacity), mVirtual(new _Ty[sizeof(_Ty) * capacity]), mCount(0)
	{
	}
	__device__ ~DeviceVector()
	{
		delete[] mVirtual;
	}
	__device__ void Add(const _Ty& data)
	{
		size_t ptr = reinterpret_cast<size_t>(mVirtual) + mOffset;
		void* casted = reinterpret_cast<void*>(ptr);

		size_t typeSize = sizeof(_Ty);
		((_Ty*)casted)[mCount] = data;
		//memcpy(casted, &data, typeSize);
		mOffset += typeSize;
		mCount++;

		return;
	}

	__device__ _Ty* GetData()
	{
		return (_Ty*)mVirtual;
	}

	__device__ void Clear()
	{
		memset(mVirtual, 0xCD, mOffset);
		mOffset = 0;
		mCount = 0;
	}

	__device__ size_t GetCount() const
	{
		return mCount;
	}

private:
	void* mVirtual;
	size_t mOffset;
	size_t mSize;
	size_t mCount;
};