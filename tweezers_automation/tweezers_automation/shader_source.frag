#version 300

const vec4 white = vec4(1,1,1,1);
int n;                     //number of spots

float totalA;         //total of the "intensity" parameters (used for intensity shaping)

vec4 spots[200]; //spot parameters- each spot corresponds to 4 vec4, first one is x,y,z,l, second one is amplitude, -,-,-

                                            //element 0 x  y  z  l    (x,y,z in um and l is an integer)
                                            //element 1 intensity (I) phase -  -
                                            //element 2 na.x na.y na.r -  (the x, y position, and radius, of the spot on the SLM- useful for Shack-Hartmann holograms)
                                            //element 3 line trapping x y z and phase gradient.  xyz define the size and angle of the line, phase gradient (between +/-1) is the
                                            //scattering force component along the line.  Zero is usually a good choice for in-plane line traps
vec2 centre;        //=vec2(0.5,0.5);//centre of the hologram as a fraction of its size (usually 0.5,0.5)

vec2 size;            //=vec2(7000,7000);//size of the hologram in microns

float f;                 //=1600; //focal length in microns

float k;                //=9.36; //wavevector in 1/microns

float blazing[32]; //blazing function

uniform float zernikeCoefficients[12]; //zernike coefficients, matching the modes defined below
uniform vec3 zernx;        //=vec3(0.0,0.0,0.0);
uniform vec3 zerny;        //=vec3(0.0,0.0,0.0);
uniform vec3 zernz;        //=vec3(0.0,0.0,0.0);


const float pi = 3.1415;

float wrap2pi(float phase){
  return mod(phase + pi, 2.0*pi) -pi;
}

float phase_to_gray(float phase){
  return phase/2.0/pi +0.5;
}

vec2 unitvector(float angle){
  return vec2(cos(angle), sin(angle));
}

float apply_LUT(float phase){
  int phint = int(floor((phase/2.0/pi +0.5)*30.9999999)); //blazing table element just before our point

  float alpha = fract((phase/2.0/pi +0.5)*30.9999999); //remainder

  return mix(blazing[phint], blazing[phint+1], alpha); //this uses the blazing table with linear interpolation
}

vec4 gray_to_16bit(float gray){
  return vec4(fract(gray * 255.9999), floor(gray * 255.9999)/255.0, 0.0, 1.0);
}

vec4 spot(int i, int j){
  return spots[4*i +j];
}

void main(void)

{

    n = 2;
    totalA = 1.0f;
    spots[0] = vec4(0.749, 0.3529, 0.3529, 2.0);
    spots[1] = vec4(500.749, 800.3529, 0.3529, 2.0);
    spots[2] = vec4(500.749, 800.3529, 0.3529, 2.0);
    spots[3] = vec4(0.0, 0.0, 0.0, 2.0);

    spots[4] = vec4(500.00, 600.00, 0.3529, 2.0);
    spots[5] = vec4(500.749, 800.3529, 0.3529, 2.0);
    spots[6] = vec4(500.749, 800.3529, 0.3529, 2.0);
    spots[7] = vec4(0.0, 0.0, 0.0, 2.0);

    centre = vec2(0.5, 0.5);
    size = vec2(7000, 7000);
    f = 1600.0f;
    k = 9.36f;

    for (int i = 0; i < 32; i++) {
        blazing[i] = 5.0f;
    }

    blazing[0] = 3.0f;
    blazing[1] = 10.0f;

   float phase;                                                       //phase of current pixel, due to the spot (in loop) or of the resultant hologram (after loop)

   float amplitude;                                               //ditto for amplitude

   //vec2 xy=(gl_TexCoord[0].xy-centre)*size;    //current xy position in the hologram, in microns
   vec2 xy=(0.6-centre)*size;    //current xy position in the hologram, in microns

   float phi=atan(xy.x,xy.y);		                              //angle of the line joining our point to the centre of the pattern

   float length;                                                     //length of a line

   vec4 pos=vec4(xy/f,1.0-dot(xy,xy)/2.0/f/f,phi/k);

   vec4 na;                                                           //to be used later, inside the loop

	 float sx;


                                                     //this loop goes through the spots and calculates the contribution from each one, summing
                                                                            //real and imaginary parts as we go.

   vec2 total = vec2(0.0,0.0);                             //real and imaginary parts of the complex hologram for this pixel
   for(int i=0; i<n; i++){
      amplitude=spot(i,1)[0];                             //amplitude of current spot

      phase=k*dot(spot(i,0),pos)+spot(i,1)[1]; 
                                                                            //this is the basic gratings and lenses algorithm; phase=kx*x+ky*y+kz*(x^2+y^2)+l*theta

      na = spot(i,2);                                           //restrict the spot to a region of the back aperture which is na[2] in radius, centred on na.xy

      if(dot(na.xy-xy/size,na.xy-xy/size) > na[2]*na[2]){
        amplitude = 0.0;
      }
//creates an xyz line trap, needs amplitude shaping.

      vec4 line = spot(i,3);
      length=sqrt(dot(line.xyz,line.xyz));

      if(length>0.0){

		      sx=k*dot(vec4(pos.xyz,1.0*length),line);

		      if(sx!=0.0) amplitude*=sin(sx)/sx;

      }

      total += amplitude * unitvector(phase); //finally, convert from amplitude+phase to real+imaginary for the summation

   }
   amplitude = dot(total, total);

   phase=atan(total.y,total.x);

   //phase += zernikeAberration(); //apply aberration correction
   if(amplitude==0.0) phase=0.0;                                      //don't focus zero order

   if(totalA>0.0){ //do amplitude-shaping (dumps light into zero order when not needed)

     phase *= clamp(amplitude/totalA,0.0,1.0);

   }

   phase = wrap2pi(phase);

//   gl_FragColor = gray_to_16bit( apply_LUT(phase)); //16-bit output with LUT, for 16 bit BNS modulators
   gl_FragColor = vec4(0.5,1,0.5,1) * apply_LUT(phase);  //8-bit output with LUT, best for Hamamatsu/Holoeye (works for BNS too)
//   gl_FragColor = vec4(1,1,1,1) * phase_to_gray(phase);          //8-bit output, linear LUT, mostly here for debug purposes
//   gl_FragColor = clamp(vec4(.866*cos(phase) -.5*sin(phase), -.866*cos(phase)-.5*sin(phase),sin(phase),1.0)/1.5+0.6667,0.0,1.0); //phase rainbow output
//   gl_FragColor = vec4(0.8, 0.9, 0.7, 1.0);

}
