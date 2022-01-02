#pragma once

struct DeviceEntity
{
	friend struct DeviceMemory;

	DeviceEntity(void* virtualPtr, size_t size);
	virtual ~DeviceEntity();

	void* GetVirtual() const { return mVirtual; }
private:
	void* mVirtual;
	size_t mSize;
};