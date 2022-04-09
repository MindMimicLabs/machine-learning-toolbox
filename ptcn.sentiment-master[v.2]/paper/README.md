> Each journal will have its own particular format.
> This repository makes no judgements, but at the same time recommends kniting to PDF first.
> There _will_ be formatting weirdness especially around images not quite being in the correct place.
> **Ignore these.**
> There will be plenty of time to move things up or down a little _after_ all the other work is done. 
> Do this so you know both if the document won't knot at all _and_ defer all the arguments around positioning untill it is really necessary.
> A lot of information on creating a properly formatted PFD can be found [here](https://bookdown.org/yihui/bookdown/latexpdf.html) and [here](https://cran.r-project.org/web/packages/rticles/index.html).

Below can be found the necessary steps to re-generate the paper from the raw `.rmd`s.
If re-generation of the results is also desired, please refer to the instruction in the [code readme](../code/README.md).
All the results, tables, and figures should be stored in the `./results` folder so the regeneration of the paper should be very fast.

## Steps

1. Double click on `index.rmd file`
2. Click knit

## Packages

> The packages listed here should be limited to the ones needed to knit the paper, not ones you used in the analysis.
> The boiler plate ones below are usually enough.

The paper itself needs `bookdown` and `kableExtra` in addtion to the packages used in the analysis.
Install them as below.

```{r}
install.packages('rmarkdown')
install.packages('kableExtra')
install.packages('bookdown')
install.packages('tinytex')

tinytex::install_tinytex()
```
