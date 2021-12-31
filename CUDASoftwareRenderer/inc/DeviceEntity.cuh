#pragma once

struct DeviceEntity
{
	friend struct DeviceMemory;

	DeviceEntity(void* virtualPtr, unsigned int size);

	void* GetVirtual() const { return mVirtual; }
private:
	void* mVirtual;
	unsigned int mSize;
};