#pragma once
struct DIB;
struct DeviceTexture;
struct ColorRGBA;
struct ResourceManager;

struct Renderer
{
public:
	Renderer(std::shared_ptr<DIB> dib, std::unique_ptr<ResourceManager>&& rs);

	void ClearCanvas(ColorRGBA clearColor);

	void Present();
private:
	std::shared_ptr<DIB> mCanvas;

	std::shared_ptr<DeviceTexture> mBuffer;
};