#include <alsa/asoundlib.h>
#define PERIOD 256 //this is the PERIOD for the ALSA buffers to call the hardware interrupt
#define SAMPLE_RATE 48000
#define INPUT_FREQUENCY 20000

int printStreamTypes() {
  int val;

  printf("ALSA library version: %s\n",
          SND_LIB_VERSION_STR);

  printf("\nPCM stream types:\n");
  for (val = 0; val <= SND_PCM_STREAM_LAST; val++)
    printf("  %s\n",
      snd_pcm_stream_name((snd_pcm_stream_t)val));

  printf("\nPCM access types:\n");
  for (val = 0; val <= SND_PCM_ACCESS_LAST; val++)
    printf("  %s\n",
      snd_pcm_access_name((snd_pcm_access_t)val));

  printf("\nPCM formats:\n");
  for (val = 0; val <= SND_PCM_FORMAT_LAST; val++)
    if (snd_pcm_format_name((snd_pcm_format_t)val)
      != NULL)
      printf("  %s (%s)\n",
        snd_pcm_format_name((snd_pcm_format_t)val),
        snd_pcm_format_description(
                           (snd_pcm_format_t)val));

  printf("\nPCM subformats:\n");
  for (val = 0; val <= SND_PCM_SUBFORMAT_LAST;
       val++)
    printf("  %s (%s)\n",
      snd_pcm_subformat_name((
        snd_pcm_subformat_t)val),
      snd_pcm_subformat_description((
        snd_pcm_subformat_t)val));

  printf("\nPCM states:\n");
  for (val = 0; val <= SND_PCM_STATE_LAST; val++)
    printf("  %s\n",
           snd_pcm_state_name((snd_pcm_state_t)val));

  return 0;
}


int simpleCapture() {

    int rc;
    snd_pcm_t *handle;
    snd_pcm_hw_params_t *params;
    int buffer[PERIOD*2]; //how large 1 buffer will be - should hold 2-3 periods
    snd_pcm_uframes_t framesWanted;
    int samplingRate = SAMPLE_RATE;
    snd_pcm_uframes_t periodSize = PERIOD; //how large 1 buffer period will be, in number of frames (samples)
    int dir;
    //open stream for recording
    rc = snd_pcm_open(&handle, "hw:0,1", SND_PCM_STREAM_CAPTURE, 0); //KEY: hw01 is the mic adc on the vm audio input enabled linux machine

    //set hardware parameters using all the relevant methods

    snd_pcm_hw_params_alloca(&params);

    rc = snd_pcm_hw_params(handle, params);
    
    // fill with default values
    snd_pcm_hw_params_any(handle, params);

    //set period size
    snd_pcm_hw_params_set_period_size_near(handle, params, &periodSize, &dir);

    snd_pcm_hw_params(handle, params);

    //test loop for playing 5 seconds of data
      //5 seconds of data = (48000 samples per second) * (5 seconds) / 256 
      
    int loop = 960; //960 interrupts plays roughly 5 seconds of data
    while (loop--) {

      //receive the signal into my own buffer
      //in BLOCKING mode (by default): waits until buffer full
      rc = snd_pcm_readi(handle, buffer, periodSize);
      
      printf("%d", *buffer); //print first value of buffer
    }

    snd_pcm_drain(handle);
    snd_pcm_close(handle);

    return 0;

}

int main() {

    int rc = simpleCapture();

    return rc;

}