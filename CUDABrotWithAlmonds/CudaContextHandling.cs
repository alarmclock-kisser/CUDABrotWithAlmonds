
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace CUDABrotWithAlmonds
{
	public class CudaContextHandling
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private string Repopath;
		private ListBox LogList;
		private ComboBox DevicesCombo;
		public ComboBox KernelsCombo;

		public int Index = -1;
		public CUdevice? Device = null;
		public PrimaryContext? Context = null;

		public CudaMemoryHandling? MemoryH;
		public CudaKernelHandling? KernelH;

		// ----- ----- CONSTRUCTORS ----- ----- \\
		public CudaContextHandling(string repopath, ListBox listBox_log, ComboBox comboBox_devices, ComboBox comboBox_kernels)
		{
			this.Repopath = repopath;
			this.LogList = listBox_log;
			this.DevicesCombo = comboBox_devices;
			this.KernelsCombo = comboBox_kernels;

			// Fill devices combobox
			this.FillDevicesCombobox();

			// Register events
			this.DevicesCombo.SelectedIndexChanged += (s, e) => this.InitDevice(this.DevicesCombo.SelectedIndex);

		}




		// ----- ----- METHODS ----- ----- \\
		public string Log(string message = "", string inner = "", int indent = 0)
		{
			string indentString = new string(' ', indent);
			string logMessage = $"[Ctx] {indentString}{message} ({inner})";
			this.LogList.Items.Add(logMessage);
			this.LogList.TopIndex = this.LogList.Items.Count - 1;
			return logMessage;
		}


		public int GetDeviceCount()
		{
			int deviceCount = CudaContext.GetDeviceCount();
			return deviceCount;
		}

		public List<CUdevice> GetDevices()
		{
			List<CUdevice> devices = [];
			int deviceCount = this.GetDeviceCount();

			for (int i = 0; i < deviceCount; i++)
			{
				CUdevice device = new CUdevice(i);
				devices.Add(device);
			}

			return devices;
		}

		public Version GetCapability(int index = -1)
		{
			index = index == -1 ? this.Index : index;

			return CudaContext.GetDeviceComputeCapability(index);
		}

		public string GetName(int index = -1)
		{
			index = index == -1 ? this.Index : index;
			return CudaContext.GetDeviceName(index);
		}

		public void FillDevicesCombobox(ComboBox? comboBox = null)
		{
			comboBox ??= this.DevicesCombo;
			comboBox.Items.Clear();

			List<CUdevice> devices = this.GetDevices();
			for (int i = 0; i < devices.Count; i++)
			{
				CUdevice device = devices[i];
				string deviceName = CudaContext.GetDeviceName(i);
				Version capability = this.GetCapability(i);
				comboBox.Items.Add($"{deviceName} ({capability.Major}.{capability.Minor})");
			}

			comboBox.SelectedIndex = this.Index;
		}

		public void InitDevice(int index = -1)
		{
			this.Dispose();

			index = index == -1 ? this.Index : index;
			if (index < 0 || index >= this.GetDeviceCount())
			{
				this.Log("Invalid device index", "Index out of range");
				return;
			}			

			this.Index = index;
			this.Device = new CUdevice(index);
			this.Context = new PrimaryContext(this.Device.Value);
			this.Context.SetCurrent();
			this.MemoryH = new CudaMemoryHandling(this.Repopath, this.LogList, this.Context);
			this.KernelH = new CudaKernelHandling(this.Repopath, this.LogList, this.Context, this.MemoryH, this.KernelsCombo);

			this.Log($"Device {index} initialized", this.GetName().Split(' ').FirstOrDefault() ?? "N/A");

		}

		public void Dispose()
		{
			this.Context?.Dispose();
			this.Context = null;
			this.Device = null;
			this.MemoryH?.Dispose();
			this.MemoryH = null;
			this.KernelH?.Dispose();
			this.KernelH = null;
		}

	}
}