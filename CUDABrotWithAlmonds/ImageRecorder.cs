using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDABrotWithAlmonds
{
	public class ImageRecorder
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		public string Repopath;
		public Label CountLabel;


		public List<System.Drawing.Image> CachedImages = [];
		public List<long> CachedIntervalls = [];



		// ----- ----- CONSTRUCTORS ----- ----- \\
		public ImageRecorder(string repopath, Label? countLabel = null)
		{
			// Set attributes
			this.Repopath = repopath;
			this.CountLabel = countLabel ?? new Label();

			// Reset cache
			this.ResetCache();
		}




		// ----- ----- METHODS ----- ----- \\
		public void ResetCache()
		{
			this.CachedImages.Clear();
			this.CachedIntervalls.Clear();

			// GC
			GC.Collect();
			GC.WaitForPendingFinalizers();

			// Update label
			this.CountLabel.Text = $"Images: -";
		}


		public void AddImage(System.Drawing.Image image, long interval)
		{
			this.CachedImages.Add(image);
			this.CachedIntervalls.Add(interval);

			// Update label
			this.CountLabel.Text = $"Images: {this.CachedImages.Count}";
		}






	}
}
