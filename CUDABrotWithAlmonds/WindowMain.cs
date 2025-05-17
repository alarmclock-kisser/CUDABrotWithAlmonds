using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CUDABrotWithAlmonds
{
	public partial class WindowMain : Form
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		public string Repopath;

		private ImageHandling ImageH;
		private ImageRecorder Recorder;

		private CudaContextHandling ContextH;

		private GuiBuilder GuiB;



		private bool MandelbrotMode = false;
		private bool isDragging = false;
		private Point mouseDownLocation = new(0, 0);
		private bool ctrlKeyPressed;
		private bool kernelExecutionRequired;
		private float mandelbrotZoomFactor = 1.0f;
		private Stopwatch stopwatch = new();

		// ----- ----- CONSTRUCTORS ----- ----- \\
		public WindowMain()
		{
			this.InitializeComponent();

			// Set repopath
			this.Repopath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\"));

			// Window position
			this.StartPosition = FormStartPosition.Manual;
			this.Location = new Point(0, 0);

			// Init. classes
			this.Recorder = new ImageRecorder(this.Repopath, this.label_cached);
			this.ImageH = new ImageHandling(this.Repopath, this.listBox_images, this.pictureBox_view, null, this.label_meta);
			this.ContextH = new CudaContextHandling(this.Repopath, this.listBox_log, this.comboBox_devices, this.comboBox_kernels);
			this.GuiB = new GuiBuilder(this.Repopath, this.listBox_log, this.ContextH, this.ImageH, this.panel_kernel);

			// Register events
			this.listBox_images.DoubleClick += (s, e) => this.MoveImage(this.listBox_images.SelectedIndex);
			this.listBox_log.MouseHover += (s, e) => this.ShowLogTooltip();
			this.comboBox_kernels.SelectedIndexChanged += (s, e) => this.LoadKernel(this.comboBox_kernels.SelectedItem?.ToString() ?? "");

			// Select first device
			if (this.comboBox_devices.Items.Count > 0)
			{
				this.comboBox_devices.SelectedIndex = 0;
			}

			// Set latest kernel
			// this.ContextH.KernelH?.SelectLatestKernel();
		}





		// ----- ----- METHODS ----- ----- \\
		public void ShowLogTooltip(int index = -1)
		{
			if (index == -1 && this.listBox_log.SelectedIndex != -1)
			{
				index = this.listBox_log.SelectedIndex;
			}
			if (index < 0 || index >= this.listBox_log.Items.Count)
			{
				return;
			}
			string message = this.listBox_log.Items[index].ToString() ?? "N/A";
			ToolTip tooltip = new();
			tooltip.Show(message, this.listBox_log, 0, 0, 10000);
		}

		public void MoveImage(int index = -1, bool refresh = true)
		{
			if (index == -1 && this.ImageH.CurrentObject != null)
			{
				index = this.ImageH.Images.IndexOf(this.ImageH.CurrentObject);
			}

			if (index < 0 && index >= this.ImageH.Images.Count)
			{
				MessageBox.Show("Invalid index", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			ImageObject image = this.ImageH.Images[index];

			// Move image Host <-> CUDA
			if (image.OnHost)
			{
				// Move to CUDA: Get bytes
				byte[] bytes = image.GetPixelsAsBytes();

				// Create buffer
				IntPtr pointer = this.ContextH.MemoryH?.PushData(bytes) ?? 0;

				// Check pointer
				if (pointer == 0)
				{
					MessageBox.Show("Failed to push data to CUDA", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}

				// Set pointer, void image
				image.Pointer = pointer;
				image.Img = null;
			}
			else if (image.OnDevice)
			{
				// Move to Host
				byte[] bytes = this.ContextH.MemoryH?.PullData<byte>(image.Pointer, true) ?? [];
				if (bytes.Length == 0)
				{
					MessageBox.Show("Failed to pull data from CUDA", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}

				// Create image
				image.SetImageFromBytes(bytes, true);
			}

			// Refill list
			if (refresh)
			{
				this.ImageH.FillImagesListBox();
			}
		}

		public void LoadKernel(string kernelName = "")
		{
			// Load kernel
			this.ContextH.KernelH?.LoadKernel(kernelName);
			if (this.ContextH.KernelH?.Kernel == null)
			{
				MessageBox.Show("Failed to load kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Toggle mandelbrot mode
			if (kernelName.ToLower().Contains("mandelbrot"))
			{
				this.ToggleMandelbrotMode(true);
			}
			else
			{
				this.ToggleMandelbrotMode(false);
			}

			// Get arguments
			this.GuiB.BuildPanel();

		}

		public void ExecuteKernelOOP(int index = -1, string kernelName = "", bool addMod = false)
		{
			// If index is -1, use current object
			if (index == -1 && this.ImageH.CurrentObject != null)
			{
				index = this.ImageH.Images.IndexOf(this.ImageH.CurrentObject);
			}

			if (index < 0 && index >= this.ImageH.Images.Count)
			{
				MessageBox.Show("Invalid index", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Get image
			ImageObject? image = this.ImageH.Images[index];
			if (image == null)
			{
				MessageBox.Show("Invalid image", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// STOPWATCH
			Stopwatch sw = Stopwatch.StartNew();

			// Verify image on device
			bool moved = false;
			if (image.OnHost)
			{
				this.MoveImage(index);
				moved = true;
				if (image.OnHost)
				{
					MessageBox.Show("Could not move image to device", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return;
				}
			}

			// Load kernel
			this.ContextH.KernelH?.LoadKernel(kernelName);
			if (this.ContextH.KernelH?.Kernel == null)
			{
				MessageBox.Show("Failed to load kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// If kernel is Mandelbrot, set mode
			if (kernelName.ToLower().Contains("mandelbrot"))
			{
				this.ToggleMandelbrotMode(true);
			}
			else
			{
				this.ToggleMandelbrotMode(false);
			}

			// Get image attributes for kernel call
			IntPtr pointer = image.Pointer;
			int width = image.Width;
			int height = image.Height;
			int channels = 4;
			int bitdepth = image.BitsPerPixel / channels;

			// Get variable arguments
			object[] args = this.GuiB.GetArgumentValues();

			// Call exec kernel -> pointer
			image.Pointer = this.ContextH.KernelH?.ExecuteKernel(pointer, width, height, channels, bitdepth, args) ?? image.Pointer;

			// Optional: Move back to host
			if (moved)
			{
				this.MoveImage(index);
			}

			// STOPWATCH
			sw.Stop();
			this.label_execTime.Text = $"Execution time: {sw.ElapsedMilliseconds} ms";

			// If kernel is Mandelbrot, cache image with interval
			if (this.MandelbrotMode && image.Img != null)
			{
				this.Recorder.AddImage(image.Img, this.stopwatch.ElapsedMilliseconds);
				this.stopwatch.Restart(); // Restart stopwatch
			}

			// Add modification to image
			if (addMod)
			{
				image.AddModification(this.ContextH.KernelH?.KernelName ?? "Unknown", args.LastOrDefault()?.ToString() ?? "");
			}

			// Refill list
			this.ImageH.FillImagesListBox();
		}

		private void ToggleMandelbrotMode(bool? forceMode = null)
		{
			if (forceMode == null)
			{
				this.MandelbrotMode = !this.MandelbrotMode;
			}
			else
			{

				this.MandelbrotMode = forceMode.Value;
			}

			// 3. Alle bestehenden Event-Handler entfernen (sauberer Reset)
			this.pictureBox_view.MouseDown -= this.ImageH.ViewPBox_MouseDown;
			this.pictureBox_view.MouseMove -= this.ImageH.ViewPBox_MouseMove;
			this.pictureBox_view.MouseUp -= this.ImageH.ViewPBox_MouseUp;
			this.pictureBox_view.MouseWheel -= this.ImageH.ViewPBox_MouseWheel;

			this.pictureBox_view.MouseDown -= this.pictureBox_view_MouseDown;
			this.pictureBox_view.MouseMove -= this.pictureBox_view_MouseMove;
			this.pictureBox_view.MouseUp -= this.pictureBox_view_MouseUp;
			this.pictureBox_view.MouseWheel -= this.pictureBox_view_MouseWheel;

			// 4. Neue Event-Handler registrieren
			if (this.MandelbrotMode)
			{
				this.stopwatch = Stopwatch.StartNew(); // Start stopwatch
				this.pictureBox_view.MouseDown += this.pictureBox_view_MouseDown;
				this.pictureBox_view.MouseMove += this.pictureBox_view_MouseMove;
				this.pictureBox_view.MouseUp += this.pictureBox_view_MouseUp;
				this.pictureBox_view.MouseWheel += this.pictureBox_view_MouseWheel;
			}
			else
			{
				this.Recorder.ResetCache(); // Reset cache
				this.stopwatch.Stop(); // Stop stopwatch
				this.pictureBox_view.MouseDown += this.ImageH.ViewPBox_MouseDown;
				this.pictureBox_view.MouseMove += this.ImageH.ViewPBox_MouseMove;
				this.pictureBox_view.MouseUp += this.ImageH.ViewPBox_MouseUp;
				this.pictureBox_view.MouseWheel += this.ImageH.ViewPBox_MouseWheel;
			}
		}

		public void ExecuteKernelOOPAll(string kernelName = "")
		{
			int count = this.ImageH.Images.Count;
			if (count == 0)
			{
				MessageBox.Show("No images to process", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Load kernel
			this.ContextH.KernelH?.LoadKernel(kernelName);
			if (this.ContextH.KernelH?.Kernel == null)
			{
				MessageBox.Show("Failed to load kernel", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Move all images to device
			for (int i = 0; i < count; i++)
			{
				this.MoveImage(i, false);
			}

			// Execute kernel on all images
			for (int i = 0; i < count; i++)
			{
				this.ExecuteKernelOOP(i, kernelName);
			}

			// Move all images to host
			for (int i = 0; i < count; i++)
			{
				this.MoveImage(i, false);
			}

			// Refill list
			this.ImageH.FillImagesListBox();
		}

		public void ExportAllGif()
		{
			List<Image> images = this.ImageH.Images.Where(x => x.Img != null).Select(x => x.Img ?? new Bitmap(1, 1)).ToList();
			if (images.Count == 0)
			{
				MessageBox.Show("No images to export", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			this.Recorder.ResetCache(); // Reset cache
			this.Recorder.CachedImages = images;
			this.Recorder.CountLabel.Text = $"Images: {this.Recorder.CachedImages.Count}";

			// Create GIF
			this.button_createGif_Click(this, EventArgs.Empty);
		}


		// Mandelbrot events
		private void pictureBox_view_MouseDown(object? sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Left)
			{
				this.isDragging = true;
				this.mouseDownLocation = e.Location;
			}
		}

		private void pictureBox_view_MouseMove(object? sender, MouseEventArgs e)
		{
			if (!this.isDragging || this.ImageH.CurrentObject == null)
			{
				return;
			}

			try
			{
				// 1. Find NumericUpDown controls more efficiently
				NumericUpDown? numericX = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("offsetx"));
				NumericUpDown? numericY = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("offsety"));
				NumericUpDown? numericZ = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("zoom"));
				NumericUpDown? numericMouseX = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("mousex"));
				NumericUpDown? numericMouseY = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("mousey"));

				if (!(numericX == null || numericY == null || numericZ == null))
				{
					// 2. Calculate smooth delta with sensitivity factor and zoom
					float sensitivity = 0.001f * (float) (1 / numericZ.Value);
					decimal deltaX = (decimal) ((e.Location.X - this.mouseDownLocation.X) * -sensitivity);
					decimal deltaY = (decimal) ((e.Location.Y - this.mouseDownLocation.Y) * -sensitivity);

					// 3. Update values with boundary checks
					this.UpdateNumericValue(numericX, deltaX);
					this.UpdateNumericValue(numericY, deltaY);
				}

				// 4. Update mouse position for smoother continuous dragging
				this.mouseDownLocation = e.Location;

				// 5. Update mouse coordinates in NumericUpDown controls
				if (!(numericMouseX == null || numericMouseY == null))
				{
					this.UpdateNumericValue(numericMouseX, e.Location.X);
					this.UpdateNumericValue(numericMouseY, e.Location.Y);
				}
			}
			catch (Exception ex)
			{
				Debug.WriteLine($"MouseMove error: {ex.Message}");
			}
		}

		private void UpdateNumericValue(NumericUpDown numeric, decimal delta)
		{
			decimal newValue = numeric.Value + delta;

			// Ensure value stays within allowed range
			if (newValue < numeric.Minimum)
			{
				newValue = numeric.Minimum;
			}

			if (newValue > numeric.Maximum)
			{
				newValue = numeric.Maximum;
			}

			numeric.Value = newValue;
		}

		private void pictureBox_view_MouseUp(object? sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Left)
			{
				this.isDragging = false;

				// Re-execute kernel
				this.button_executeOOP_Click(sender, e);
			}
		}

		private void pictureBox_view_MouseWheel(object? sender, MouseEventArgs e)
		{
			// Check for CTRL key press
			if (Control.ModifierKeys == Keys.Control)
			{
				this.ctrlKeyPressed = true;
				this.kernelExecutionRequired = true; // Set the flag
				NumericUpDown? numericI = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("iter"));
				if (numericI == null)
				{
					this.ContextH.KernelH?.Log("MaxIter control not found!", "", 3);
					return;
				}

				// Increase/Decrease maxIter
				if (e.Delta > 0)
				{
					numericI.Value += 2;
				}
				else if (e.Delta < 0)
				{
					if (numericI.Value > 0)
					{
						numericI.Value -= 2;
					}
				}
				return;
			}

			// Check if CTRL key was previously pressed
			if (this.ctrlKeyPressed)
			{
				this.ctrlKeyPressed = false; // Reset the flag
				this.kernelExecutionRequired = true;
			}

			// 1. Find NumericUpDown controls more efficiently
			NumericUpDown? numericZ = this.panel_kernel.Controls.OfType<NumericUpDown>().FirstOrDefault(x => x.Name.ToLower().Contains("zoom"));
			if (numericZ == null)
			{
				this.ContextH.KernelH?.Log("Zoom control not found!", "", 3);
				return;
			}

			// 2. Calculate zoom factor
			if (e.Delta > 0)
			{
				this.mandelbrotZoomFactor *= 1.1f;
			}
			else
			{
				this.mandelbrotZoomFactor /= 1.1f;
			}

			// 3. Update zoom value with boundary checks
			decimal newValue = (decimal) this.mandelbrotZoomFactor;
			if (newValue < numericZ.Minimum)
			{
				newValue = numericZ.Minimum;
			}
			if (newValue > numericZ.Maximum)
			{
				newValue = numericZ.Maximum;
			}
			numericZ.Value = newValue;

			// Call re-exec kernel
			this.kernelExecutionRequired = true;

			if (!this.ctrlKeyPressed && this.kernelExecutionRequired)
			{
				this.kernelExecutionRequired = false;
				this.button_executeOOP_Click(sender, e);
			}
		}





		// ----- ----- EVENT HANDLERS ----- ----- \\
		private void button_import_Click(object sender, EventArgs e)
		{
			// If CTRL down, import GIF
			if (ModifierKeys == Keys.Control)
			{
				this.ImageH.ImportGif();
				return;
			}

			this.ImageH.ImportImage();
		}

		private void button_export_Click(object sender, EventArgs e)
		{
			// If CTRL down, export GIF
			if (ModifierKeys == Keys.Control)
			{
				this.ExportAllGif();
				return;
			}

			this.ImageH.CurrentObject?.Export(true);
		}

		private void button_createImage_Click(object sender, EventArgs e)
		{
			int size = Math.Min(this.pictureBox_view.Width, this.pictureBox_view.Height);
			this.ImageH.CreateEmpty(Color.White, size, "");
			this.ImageH.FitZoom();
		}

		private void button_center_Click(object sender, EventArgs e)
		{
			// If CTRL down, fit zoom
			if (ModifierKeys == Keys.Control)
			{
				this.ImageH.CenterImage();
			}
			else
			{
				this.ImageH.FitZoom();
			}
		}

		private void button_executeOOP_Click(object? sender, EventArgs e)
		{
			// If CTRL down: Execute on all images
			if (ModifierKeys == Keys.Control)
			{
				this.ExecuteKernelOOPAll(this.comboBox_kernels.SelectedItem?.ToString() ?? "");
				return;
			}

			bool addMod = !((this.comboBox_kernels.SelectedItem?.ToString() ?? "").ToLower().Contains("mandelbrot"));

			this.ExecuteKernelOOP(-1, this.comboBox_kernels.SelectedItem?.ToString() ?? "");
		}

		private void button_reset_Click(object sender, EventArgs e)
		{
			this.ImageH.CurrentObject?.ResetImage();

			this.ImageH.FillImagesListBox();
		}

		private async void button_createGif_Click(object sender, EventArgs e)
		{
			// Folder MyPictures
			string folder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "CUDA-GIFs");
			string result;

			// FRAME RATE
			int frameRate = (int) this.numericUpDown_frameRate.Value;

			// RESIZE
			Size? resize = null;
			if ((int) this.numericUpDown_size.Value < Math.Min(this.ImageH.CurrentObject?.Width ?? 0, this.ImageH.CurrentObject?.Height ?? 0));
			{
				resize = new Size((int) this.numericUpDown_size.Value, (int) this.numericUpDown_size.Value);
			}

			// If CTRL down, async call
			if (ModifierKeys == Keys.Control)
			{
				int count = this.Recorder.CachedImages.Count;
				Stopwatch sw = Stopwatch.StartNew();

				this.GuiB.Log("Creating GIF (async) ...", "", 1);
				result = await this.Recorder.CreateGifAsync(folder, this.ImageH.CurrentObject?.Name ?? "animatedGif_", frameRate, true, resize);

				sw.Stop();
				this.GuiB.Log("GIF created (async) ", $"{(sw.ElapsedMilliseconds / count)} ms/F", 1);
			}
			else
			{
				this.GuiB.Log("Creating GIF ...", "", 1);
				result = this.Recorder.CreateGif(folder, this.ImageH.CurrentObject?.Name ?? "animatedGif_", frameRate, true);
			}

			// Msgbox
			if (string.IsNullOrEmpty(result))
			{
				MessageBox.Show("Failed to create GIF", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			else
			{
				MessageBox.Show($"GIF created: {result}\n\nSize: {(resize != null ? resize.Value : new Size(this.ImageH.CurrentObject?.Width ?? 0, this.ImageH.CurrentObject?.Height ?? 0))}\nFPS: {frameRate}", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}

			// Reload image from file
			this.ImageH.CurrentObject?.ResetImage();
		}
	}
}
