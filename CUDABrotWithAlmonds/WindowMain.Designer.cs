namespace CUDABrotWithAlmonds
{
	partial class WindowMain
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.panel_view = new Panel();
			this.pictureBox_view = new PictureBox();
			this.comboBox_devices = new ComboBox();
			this.listBox_log = new ListBox();
			this.listBox_images = new ListBox();
			this.button_createImage = new Button();
			this.button_import = new Button();
			this.button_export = new Button();
			this.label_meta = new Label();
			this.progressBar_load = new ProgressBar();
			this.button_center = new Button();
			this.panel_kernel = new Panel();
			this.comboBox_kernels = new ComboBox();
			this.button_executeOOP = new Button();
			this.button_reset = new Button();
			this.label_execTime = new Label();
			this.label_cached = new Label();
			this.button_createGif = new Button();
			this.numericUpDown_frameRate = new NumericUpDown();
			this.label_fps = new Label();
			this.numericUpDown_size = new NumericUpDown();
			this.label_resize = new Label();
			this.progressBar_vram = new ProgressBar();
			this.label_pushTime = new Label();
			this.label_pullTime = new Label();
			this.groupBox_control = new GroupBox();
			this.numericUpDown_createSize = new NumericUpDown();
			this.groupBox_gif = new GroupBox();
			this.checkBox_record = new CheckBox();
			this.groupBox_kernel = new GroupBox();
			this.button_autoFractal = new Button();
			this.checkBox_optionalArgsOnly = new CheckBox();
			this.button_addKernel = new Button();
			this.checkBox_silent = new CheckBox();
			this.checkBox_crosshair = new CheckBox();
			this.button_fullScreen = new Button();
			this.panel_view.SuspendLayout();
			((System.ComponentModel.ISupportInitialize) this.pictureBox_view).BeginInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_frameRate).BeginInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_size).BeginInit();
			this.groupBox_control.SuspendLayout();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_createSize).BeginInit();
			this.groupBox_gif.SuspendLayout();
			this.groupBox_kernel.SuspendLayout();
			this.SuspendLayout();
			// 
			// panel_view
			// 
			this.panel_view.BackColor = Color.Black;
			this.panel_view.Controls.Add(this.pictureBox_view);
			this.panel_view.Location = new Point(253, 12);
			this.panel_view.Name = "panel_view";
			this.panel_view.Size = new Size(1589, 867);
			this.panel_view.TabIndex = 0;
			// 
			// pictureBox_view
			// 
			this.pictureBox_view.BackColor = Color.White;
			this.pictureBox_view.Location = new Point(51, 42);
			this.pictureBox_view.Name = "pictureBox_view";
			this.pictureBox_view.Size = new Size(512, 512);
			this.pictureBox_view.TabIndex = 0;
			this.pictureBox_view.TabStop = false;
			// 
			// comboBox_devices
			// 
			this.comboBox_devices.FormattingEnabled = true;
			this.comboBox_devices.Location = new Point(12, 12);
			this.comboBox_devices.Name = "comboBox_devices";
			this.comboBox_devices.Size = new Size(235, 23);
			this.comboBox_devices.TabIndex = 1;
			this.comboBox_devices.Text = "Initialize CUDA device ...";
			// 
			// listBox_log
			// 
			this.listBox_log.FormattingEnabled = true;
			this.listBox_log.ItemHeight = 15;
			this.listBox_log.Location = new Point(12, 785);
			this.listBox_log.Name = "listBox_log";
			this.listBox_log.Size = new Size(235, 94);
			this.listBox_log.TabIndex = 2;
			// 
			// listBox_images
			// 
			this.listBox_images.FormattingEnabled = true;
			this.listBox_images.ItemHeight = 15;
			this.listBox_images.Location = new Point(12, 684);
			this.listBox_images.Name = "listBox_images";
			this.listBox_images.Size = new Size(235, 79);
			this.listBox_images.TabIndex = 3;
			// 
			// button_createImage
			// 
			this.button_createImage.Location = new Point(174, 22);
			this.button_createImage.Name = "button_createImage";
			this.button_createImage.Size = new Size(55, 23);
			this.button_createImage.TabIndex = 4;
			this.button_createImage.Text = "Create";
			this.button_createImage.UseVisualStyleBackColor = true;
			this.button_createImage.Click += this.button_createImage_Click;
			// 
			// button_import
			// 
			this.button_import.Location = new Point(6, 22);
			this.button_import.Name = "button_import";
			this.button_import.Size = new Size(60, 23);
			this.button_import.TabIndex = 5;
			this.button_import.Text = "Import";
			this.button_import.UseVisualStyleBackColor = true;
			this.button_import.Click += this.button_import_Click;
			// 
			// button_export
			// 
			this.button_export.Location = new Point(6, 51);
			this.button_export.Name = "button_export";
			this.button_export.Size = new Size(60, 23);
			this.button_export.TabIndex = 6;
			this.button_export.Text = "Export";
			this.button_export.UseVisualStyleBackColor = true;
			this.button_export.Click += this.button_export_Click;
			// 
			// label_meta
			// 
			this.label_meta.AutoSize = true;
			this.label_meta.Location = new Point(12, 666);
			this.label_meta.Name = "label_meta";
			this.label_meta.Size = new Size(101, 15);
			this.label_meta.TabIndex = 7;
			this.label_meta.Text = "No image loaded.";
			// 
			// progressBar_load
			// 
			this.progressBar_load.Location = new Point(12, 769);
			this.progressBar_load.Name = "progressBar_load";
			this.progressBar_load.Size = new Size(235, 10);
			this.progressBar_load.TabIndex = 8;
			// 
			// button_center
			// 
			this.button_center.Location = new Point(99, 51);
			this.button_center.Name = "button_center";
			this.button_center.Size = new Size(50, 23);
			this.button_center.TabIndex = 9;
			this.button_center.Text = "Center";
			this.button_center.UseVisualStyleBackColor = true;
			this.button_center.Click += this.button_center_Click;
			// 
			// panel_kernel
			// 
			this.panel_kernel.BackColor = Color.LightGray;
			this.panel_kernel.Location = new Point(12, 113);
			this.panel_kernel.Name = "panel_kernel";
			this.panel_kernel.Size = new Size(235, 259);
			this.panel_kernel.TabIndex = 10;
			// 
			// comboBox_kernels
			// 
			this.comboBox_kernels.FormattingEnabled = true;
			this.comboBox_kernels.Location = new Point(12, 84);
			this.comboBox_kernels.Name = "comboBox_kernels";
			this.comboBox_kernels.Size = new Size(235, 23);
			this.comboBox_kernels.TabIndex = 11;
			this.comboBox_kernels.SelectedIndexChanged += this.comboBox_kernels_SelectedIndexChanged;
			// 
			// button_executeOOP
			// 
			this.button_executeOOP.Location = new Point(6, 22);
			this.button_executeOOP.Name = "button_executeOOP";
			this.button_executeOOP.Size = new Size(75, 23);
			this.button_executeOOP.TabIndex = 12;
			this.button_executeOOP.Text = "Exec OOP";
			this.button_executeOOP.UseVisualStyleBackColor = true;
			this.button_executeOOP.Click += this.button_executeOOP_Click;
			// 
			// button_reset
			// 
			this.button_reset.Location = new Point(99, 22);
			this.button_reset.Name = "button_reset";
			this.button_reset.Size = new Size(50, 23);
			this.button_reset.TabIndex = 13;
			this.button_reset.Text = "Reset";
			this.button_reset.UseVisualStyleBackColor = true;
			this.button_reset.Click += this.button_reset_Click;
			// 
			// label_execTime
			// 
			this.label_execTime.AutoSize = true;
			this.label_execTime.Location = new Point(87, 19);
			this.label_execTime.Name = "label_execTime";
			this.label_execTime.Size = new Size(71, 15);
			this.label_execTime.TabIndex = 14;
			this.label_execTime.Text = "Exec. time: -";
			// 
			// label_cached
			// 
			this.label_cached.AutoSize = true;
			this.label_cached.Location = new Point(84, 26);
			this.label_cached.Name = "label_cached";
			this.label_cached.Size = new Size(56, 15);
			this.label_cached.TabIndex = 15;
			this.label_cached.Text = "Images: -";
			// 
			// button_createGif
			// 
			this.button_createGif.Location = new Point(179, 22);
			this.button_createGif.Name = "button_createGif";
			this.button_createGif.Size = new Size(50, 23);
			this.button_createGif.TabIndex = 16;
			this.button_createGif.Text = "-> GIF";
			this.button_createGif.UseVisualStyleBackColor = true;
			this.button_createGif.Click += this.button_createGif_Click;
			// 
			// numericUpDown_frameRate
			// 
			this.numericUpDown_frameRate.Location = new Point(179, 51);
			this.numericUpDown_frameRate.Maximum = new decimal(new int[] { 60, 0, 0, 0 });
			this.numericUpDown_frameRate.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
			this.numericUpDown_frameRate.Name = "numericUpDown_frameRate";
			this.numericUpDown_frameRate.Size = new Size(50, 23);
			this.numericUpDown_frameRate.TabIndex = 17;
			this.numericUpDown_frameRate.Value = new decimal(new int[] { 10, 0, 0, 0 });
			// 
			// label_fps
			// 
			this.label_fps.AutoSize = true;
			this.label_fps.Location = new Point(144, 53);
			this.label_fps.Name = "label_fps";
			this.label_fps.Size = new Size(29, 15);
			this.label_fps.TabIndex = 18;
			this.label_fps.Text = "FPS:";
			// 
			// numericUpDown_size
			// 
			this.numericUpDown_size.Location = new Point(88, 51);
			this.numericUpDown_size.Maximum = new decimal(new int[] { 8192, 0, 0, 0 });
			this.numericUpDown_size.Minimum = new decimal(new int[] { 128, 0, 0, 0 });
			this.numericUpDown_size.Name = "numericUpDown_size";
			this.numericUpDown_size.Size = new Size(50, 23);
			this.numericUpDown_size.TabIndex = 19;
			this.numericUpDown_size.Value = new decimal(new int[] { 1024, 0, 0, 0 });
			// 
			// label_resize
			// 
			this.label_resize.AutoSize = true;
			this.label_resize.Location = new Point(52, 53);
			this.label_resize.Name = "label_resize";
			this.label_resize.Size = new Size(30, 15);
			this.label_resize.TabIndex = 20;
			this.label_resize.Text = "Size:";
			// 
			// progressBar_vram
			// 
			this.progressBar_vram.Location = new Point(12, 41);
			this.progressBar_vram.Name = "progressBar_vram";
			this.progressBar_vram.Size = new Size(235, 12);
			this.progressBar_vram.TabIndex = 21;
			// 
			// label_pushTime
			// 
			this.label_pushTime.AutoSize = true;
			this.label_pushTime.Location = new Point(87, 34);
			this.label_pushTime.Name = "label_pushTime";
			this.label_pushTime.Size = new Size(71, 15);
			this.label_pushTime.TabIndex = 22;
			this.label_pushTime.Text = "Push time: -";
			// 
			// label_pullTime
			// 
			this.label_pullTime.AutoSize = true;
			this.label_pullTime.Location = new Point(87, 49);
			this.label_pullTime.Name = "label_pullTime";
			this.label_pullTime.Size = new Size(65, 15);
			this.label_pullTime.TabIndex = 23;
			this.label_pullTime.Text = "Pull time: -";
			// 
			// groupBox_control
			// 
			this.groupBox_control.Controls.Add(this.numericUpDown_createSize);
			this.groupBox_control.Controls.Add(this.button_import);
			this.groupBox_control.Controls.Add(this.button_export);
			this.groupBox_control.Controls.Add(this.button_center);
			this.groupBox_control.Controls.Add(this.button_reset);
			this.groupBox_control.Controls.Add(this.button_createImage);
			this.groupBox_control.Location = new Point(12, 576);
			this.groupBox_control.Name = "groupBox_control";
			this.groupBox_control.Size = new Size(235, 85);
			this.groupBox_control.TabIndex = 24;
			this.groupBox_control.TabStop = false;
			this.groupBox_control.Text = "Control";
			// 
			// numericUpDown_createSize
			// 
			this.numericUpDown_createSize.Location = new Point(174, 51);
			this.numericUpDown_createSize.Maximum = new decimal(new int[] { 8192, 0, 0, 0 });
			this.numericUpDown_createSize.Minimum = new decimal(new int[] { 64, 0, 0, 0 });
			this.numericUpDown_createSize.Name = "numericUpDown_createSize";
			this.numericUpDown_createSize.Size = new Size(55, 23);
			this.numericUpDown_createSize.TabIndex = 14;
			this.numericUpDown_createSize.Value = new decimal(new int[] { 1024, 0, 0, 0 });
			// 
			// groupBox_gif
			// 
			this.groupBox_gif.Controls.Add(this.checkBox_record);
			this.groupBox_gif.Controls.Add(this.button_createGif);
			this.groupBox_gif.Controls.Add(this.label_cached);
			this.groupBox_gif.Controls.Add(this.numericUpDown_frameRate);
			this.groupBox_gif.Controls.Add(this.label_fps);
			this.groupBox_gif.Controls.Add(this.numericUpDown_size);
			this.groupBox_gif.Controls.Add(this.label_resize);
			this.groupBox_gif.Location = new Point(12, 486);
			this.groupBox_gif.Name = "groupBox_gif";
			this.groupBox_gif.Size = new Size(235, 84);
			this.groupBox_gif.TabIndex = 25;
			this.groupBox_gif.TabStop = false;
			this.groupBox_gif.Text = "GIF";
			// 
			// checkBox_record
			// 
			this.checkBox_record.AutoSize = true;
			this.checkBox_record.Location = new Point(6, 22);
			this.checkBox_record.Name = "checkBox_record";
			this.checkBox_record.Size = new Size(68, 19);
			this.checkBox_record.TabIndex = 21;
			this.checkBox_record.Text = "Record?";
			this.checkBox_record.UseVisualStyleBackColor = true;
			// 
			// groupBox_kernel
			// 
			this.groupBox_kernel.Controls.Add(this.button_autoFractal);
			this.groupBox_kernel.Controls.Add(this.checkBox_optionalArgsOnly);
			this.groupBox_kernel.Controls.Add(this.button_addKernel);
			this.groupBox_kernel.Controls.Add(this.button_executeOOP);
			this.groupBox_kernel.Controls.Add(this.label_execTime);
			this.groupBox_kernel.Controls.Add(this.label_pushTime);
			this.groupBox_kernel.Controls.Add(this.label_pullTime);
			this.groupBox_kernel.Location = new Point(12, 378);
			this.groupBox_kernel.Name = "groupBox_kernel";
			this.groupBox_kernel.Size = new Size(235, 102);
			this.groupBox_kernel.TabIndex = 26;
			this.groupBox_kernel.TabStop = false;
			this.groupBox_kernel.Text = "Kernel";
			// 
			// button_autoFractal
			// 
			this.button_autoFractal.Location = new Point(174, 73);
			this.button_autoFractal.Name = "button_autoFractal";
			this.button_autoFractal.Size = new Size(55, 23);
			this.button_autoFractal.TabIndex = 26;
			this.button_autoFractal.Text = "Auto...";
			this.button_autoFractal.UseVisualStyleBackColor = true;
			this.button_autoFractal.Click += this.button_autoFractal_Click;
			// 
			// checkBox_optionalArgsOnly
			// 
			this.checkBox_optionalArgsOnly.AutoSize = true;
			this.checkBox_optionalArgsOnly.Checked = true;
			this.checkBox_optionalArgsOnly.CheckState = CheckState.Checked;
			this.checkBox_optionalArgsOnly.Location = new Point(6, 77);
			this.checkBox_optionalArgsOnly.Name = "checkBox_optionalArgsOnly";
			this.checkBox_optionalArgsOnly.Size = new Size(105, 19);
			this.checkBox_optionalArgsOnly.TabIndex = 25;
			this.checkBox_optionalArgsOnly.Text = "Only variables?";
			this.checkBox_optionalArgsOnly.UseVisualStyleBackColor = true;
			this.checkBox_optionalArgsOnly.CheckedChanged += this.checkBox_optionalArgsOnly_CheckedChanged;
			// 
			// button_addKernel
			// 
			this.button_addKernel.Location = new Point(6, 51);
			this.button_addKernel.Name = "button_addKernel";
			this.button_addKernel.Size = new Size(74, 23);
			this.button_addKernel.TabIndex = 24;
			this.button_addKernel.Text = "Add ...";
			this.button_addKernel.UseVisualStyleBackColor = true;
			this.button_addKernel.Click += this.button_addKernel_Click;
			// 
			// checkBox_silent
			// 
			this.checkBox_silent.AutoSize = true;
			this.checkBox_silent.Location = new Point(98, 59);
			this.checkBox_silent.Name = "checkBox_silent";
			this.checkBox_silent.Size = new Size(60, 19);
			this.checkBox_silent.TabIndex = 26;
			this.checkBox_silent.Text = "Silent?";
			this.checkBox_silent.UseVisualStyleBackColor = true;
			// 
			// checkBox_crosshair
			// 
			this.checkBox_crosshair.AutoSize = true;
			this.checkBox_crosshair.Location = new Point(12, 59);
			this.checkBox_crosshair.Name = "checkBox_crosshair";
			this.checkBox_crosshair.Size = new Size(80, 19);
			this.checkBox_crosshair.TabIndex = 27;
			this.checkBox_crosshair.Text = "Crosshair?";
			this.checkBox_crosshair.UseVisualStyleBackColor = true;
			this.checkBox_crosshair.CheckedChanged += this.checkBox_crosshair_CheckedChanged;
			// 
			// button_fullScreen
			// 
			this.button_fullScreen.Location = new Point(172, 59);
			this.button_fullScreen.Name = "button_fullScreen";
			this.button_fullScreen.Size = new Size(75, 23);
			this.button_fullScreen.TabIndex = 28;
			this.button_fullScreen.Text = "Fullscreen";
			this.button_fullScreen.UseVisualStyleBackColor = true;
			this.button_fullScreen.Click += this.button_fullScreen_Click;
			// 
			// WindowMain
			// 
			this.AutoScaleDimensions = new SizeF(7F, 15F);
			this.AutoScaleMode = AutoScaleMode.Font;
			this.ClientSize = new Size(1854, 891);
			this.Controls.Add(this.button_fullScreen);
			this.Controls.Add(this.checkBox_silent);
			this.Controls.Add(this.checkBox_crosshair);
			this.Controls.Add(this.groupBox_kernel);
			this.Controls.Add(this.groupBox_gif);
			this.Controls.Add(this.label_meta);
			this.Controls.Add(this.groupBox_control);
			this.Controls.Add(this.progressBar_vram);
			this.Controls.Add(this.comboBox_kernels);
			this.Controls.Add(this.panel_kernel);
			this.Controls.Add(this.progressBar_load);
			this.Controls.Add(this.listBox_images);
			this.Controls.Add(this.listBox_log);
			this.Controls.Add(this.comboBox_devices);
			this.Controls.Add(this.panel_view);
			this.MaximumSize = new Size(1870, 930);
			this.MinimumSize = new Size(1870, 930);
			this.Name = "WindowMain";
			this.Text = "Form1";
			this.panel_view.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize) this.pictureBox_view).EndInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_frameRate).EndInit();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_size).EndInit();
			this.groupBox_control.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_createSize).EndInit();
			this.groupBox_gif.ResumeLayout(false);
			this.groupBox_gif.PerformLayout();
			this.groupBox_kernel.ResumeLayout(false);
			this.groupBox_kernel.PerformLayout();
			this.ResumeLayout(false);
			this.PerformLayout();
		}

		#endregion

		private Panel panel_view;
		private PictureBox pictureBox_view;
		private ComboBox comboBox_devices;
		private ListBox listBox_log;
		private ListBox listBox_images;
		private Button button_createImage;
		private Button button_import;
		private Button button_export;
		private Label label_meta;
		private ProgressBar progressBar_load;
		private Button button_center;
		private Panel panel_kernel;
		private ComboBox comboBox_kernels;
		private Button button_executeOOP;
		private Button button_reset;
		private Label label_execTime;
		private Label label_cached;
		private Button button_createGif;
		private NumericUpDown numericUpDown_frameRate;
		private Label label_fps;
		private NumericUpDown numericUpDown_size;
		private Label label_resize;
		private ProgressBar progressBar_vram;
		private Label label_pushTime;
		private Label label_pullTime;
		private GroupBox groupBox_control;
		private GroupBox groupBox_gif;
		private GroupBox groupBox_kernel;
		private NumericUpDown numericUpDown_createSize;
		private CheckBox checkBox_crosshair;
		private CheckBox checkBox_record;
		private Button button_addKernel;
		private CheckBox checkBox_optionalArgsOnly;
		private CheckBox checkBox_silent;
		private Button button_autoFractal;
		private Button button_fullScreen;
	}
}