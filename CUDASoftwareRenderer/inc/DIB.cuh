#pragma once

struct DeviceTexture;
struct DIB
{
public:
	DIB(HWND hWnd, unsigned int width, unsigned int height);
	~DIB();

	unsigned int GetWidth() const { return mWidth; }
	unsigned int GetHeight() const { return mHeight; }

	DWORD* GetRaw() const { return mRaw; }

	void CopyBuffer(std::shared_ptr<DeviceTexture> virtualResource);
	void Present();

private:
	unsigned int mWidth;
	unsigned int mHeight;

	HDC mMemoryDC;
	HDC mHandleDC;

	HBITMAP mBitmap;
	DWORD* mRaw;

	HWND mHandle;
};