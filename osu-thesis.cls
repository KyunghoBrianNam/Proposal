\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{osu-thesis}[2021/1/19 v1.0 Oklahoma State University thesis]
\LoadClass[letterpaper,oneside,11pt]{memoir}

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[left=1.5in,right=1in,top=1in,bottom=1.4in,headheight=23pt,headsep=13pt,footskip=.5in]{geometry}
\RequirePackage[backend=biber,style=authoryear,sorting=nty,hyperref=true]{biblatex}
\RequirePackage[american]{babel}
\RequirePackage[babel=true]{csquotes}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{url}
\RequirePackage{graphicx}
\RequirePackage{etoolbox}
\RequirePackage{enumitem}
\RequirePackage{xcolor}
\RequirePackage{listings}
\RequirePackage[chapter]{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage[font=normalsize,labelsep=quad,labelfont=bf]{caption}
\RequirePackage{calc}
\RequirePackage{booktabs}
\RequirePackage{amsfonts}

\setlength\intextsep{26pt}
\setlength{\textfloatsep}{26pt}
\setlength{\floatsep}{26pt}
\setlength{\abovecaptionskip}{13pt}
\captionsetup[table]{belowskip=0pt,aboveskip=13pt}
\captionsetup[algorithm]{labelsep=quad}

\DeclareFixedFont{\lstnormal}{T1}{SourceCodePro-TLF}{m}{n}{10}
\DeclareFixedFont{\lstboldface}{T1}{SourceCodePro-TLF}{sb}{n}{10}
\DeclareFixedFont{\lstslanted}{T1}{SourceCodePro-TLF}{m}{sl}{10}
\DeclareFixedFont{\lstboldslanted}{T1}{SourceCodePro-TLF}{sb}{sl}{10}
\newlength{\lstem}
\settowidth{\lstem}{\lstnormal M}%
\definecolor{lstGrey}{RGB}{128, 128, 128}
\definecolor{lstRed}{RGB}{152,0,0}
\definecolor{lstOrange}{RGB}{166, 112, 25}
\definecolor{lstGold}{RGB}{166, 154, 25}
\definecolor{lstGreen}{RGB}{12,80,0}
\definecolor{lstTeal}{RGB}{8, 103, 112}
\definecolor{lstBlue}{RGB}{0,0,144}
\definecolor{lstLilac}{RGB}{83, 8, 112}
\definecolor{lstFuscia}{RGB}{112, 8, 72}
\colorlet{lstOperator}{lstBlue}

\lstloadlanguages{[Sharp]C,Python,R}
\lstset{%
  abovecaptionskip=0pt,%
  belowcaptionskip=3pt,%
  numbers=left,%
  numbersep=1em,%
  numberstyle=\tiny,%
  firstnumber=1,%
  stepnumber=5,%
  frame=single,%
  framesep=\fboxsep,%
  framerule=\fboxrule,%
  frameround={t}{t}{t}{t},%
  rulecolor=\color{black},%
  xleftmargin=\dimexpr\fboxsep+\fboxrule\relax,%
  xrightmargin=\dimexpr\fboxsep+\fboxrule\relax,%
  columns=flexible,%
  breaklines=true,%
  breakatwhitespace=true,%
  breakindent=8\lstem,%
  postbreak={\makebox[3\lstem]{\hfill\textcolor{lstGrey}{\(\hookrightarrow\)}\space}},%
  keywordstyle=[1]\color{lstFuscia}\lstboldslanted,% Flow-control reserved words 
  keywordstyle=[2]\color{lstBlue}\lstboldface,% Other reserved words
  keywordstyle=[3]\color{lstLilac},% Special literals
  keywordstyle=[4]\color{lstFuscia},% Built-in types
  keywordstyle=[5]\color{lstBlue},% Built-in functions
  keywordstyle=[6]\color{lstOperator},% Operators
  keywordstyle=[7]\color{lstOrange},% Library 1
  keywordstyle=[8]\color{lstTeal},% Library 2
  keywordstyle=[9]\color{lstGold},% Library 3
  basicstyle=\lstnormal,%
  stringstyle=\color{lstRed},%
  commentstyle=\color{lstGreen}\lstslanted,%
  identifierstyle=\color{black},%
  keepspaces=true,%
  showspaces=false,%
  showstringspaces=false,%
  tabsize=3%
}

\lstdefinelanguage{PythonPlus}{%
  language=Python,%
  morestring=[s]{f'}{'},%
  morestring=[s]{f"}{"},%
  morestring=[s]{f'''}{'''},%
  morestring=[s]{f"""}{"""},%
  literate=*% Operators
    {[}{\textcolor{lstOperator}{[}}{1}%
    {]}{\textcolor{lstOperator}{]}}{1}%
    {:}{\textcolor{lstOperator}{:}}{1}%
    {!}{\textcolor{lstOperator}{!}}{1}%
    {.}{\textcolor{lstOperator}{.}}{1}%
    {@}{\textcolor{lstOperator}{@}}{1}%
    {=}{\textcolor{lstOperator}{=}}{1}%
    {<}{\textcolor{lstOperator}{<}}{1}%
    {>}{\textcolor{lstOperator}{>}}{1}%
    {+}{\textcolor{lstOperator}{+}}{1}%
    {-}{\textcolor{lstOperator}{-}}{1}%
    {*}{\textcolor{lstOperator}{*}}{1}%
    {/}{\textcolor{lstOperator}{/}}{1}%
    {\{}{\textcolor{lstOperator}{\{}}{1}%
    {\}}{\textcolor{lstOperator}{\}}}{1}%
    {\&}{\textcolor{lstOperator}{\&}}{1}%
    {\%}{\textcolor{lstOperator}{\%}}{1}%
    {\^}{\textcolor{lstOperator}{\^}}{1}%
    {\|}{\textcolor{lstOperator}{\|}}{1},%
  keywords=[1]{% Flow-control reserved words
    assert,async,await,break,continue,elif,else,except,%
    finally,for,if,raise,return,try,while,with,yield%
  },%
  keywords=[2]{% Other reserved keywords
    as,class,def,del,from,global,import,in,is,lambda,nonlocal,pass%
  },%
  keywords=[3]{% Special literals
    False,None,True%
  },%
  keywords=[4]{% Built-in Types
    bool,bytearray,bytes,complex,dict,float,frozenset,int,list,memoryview,%
    object,range,set,slice,str,tuple%
  },%
  keywords=[5]{% Built-in functions
    __import__,abs,all,any,ascii,bin,breakpoint,callable,chr,classmethod,%
    compile,delattr,dir,divmod,enumerate,eval,exec,filter,format,getattr,%
    globals,hasattr,hash,help,hex,id,input,isinstance,issubclass,iter,len,%
    locals,map,max,min,next,oct,open,ord,pow,print,property,range,repr,%
    reversed,round,setattr,sorted,staticmethod,sum,super,type,vars,zip%
  },%
  keywords=[6]{% Operator keywords
    and,not,or%
  },%
  keywords=[7]{% numpy
    arange,arccos,arccosh,arcsin,arcsinh,arctan,arctan2,arctanh,around,%
    cla,column_stack,concatenate,cos,cosh,cumprod,cumsum,%
    deg2rad,degrees,det,diff,dot,dtype,eig,eigs,eigvals,empty,exp,eye,%
    find,fix,flatten,floor,fsolve,grid,hstack,hypot,isscalar,legend,%
    linspace,logspace,lstsq,mean,ndarray,norm,numpy.char,numpy.ma,numpy.rec,ode,%
    ones,pcolor,pi,plot,polyfit,polyval,qr,quad,rad2deg,radians,%
    rand,reshape,rint,roll,round_,shape,sin,sinh,solve,sqrt,squeeze,%
    svd,tan,tanh,unwrap,vander,vectorize,vstack,xlabel,%
    ylabel,zeros%
  },%
  keywords=[7]{%
  },%
  keywords=[8]{%
  },%
  keywords=[9]{%
  }%
}

\lstdefinelanguage{RPlus}{%
  language=R,%
  sensitive=true,%
  otherkeywords=,%
  alsoother={$},%
  alsoletter={.},%
  literate=*%
    {[}{\textcolor{lstOperator}{[}}{1}%
    {]}{\textcolor{lstOperator}{]}}{1}%
    {:}{\textcolor{lstOperator}{:}}{1}%
    {@}{\textcolor{lstOperator}{@}}{1}%
    {!}{\textcolor{lstOperator}{!}}{1}%
    {~}{\textcolor{lstOperator}{~}}{1}%
    {=}{\textcolor{lstOperator}{=}}{1}%
    {<}{\textcolor{lstOperator}{<}}{1}%
    {>}{\textcolor{lstOperator}{>}}{1}%
    {+}{\textcolor{lstOperator}{+}}{1}%
    {-}{\textcolor{lstOperator}{-}}{1}%
    {*}{\textcolor{lstOperator}{*}}{1}%
    {/}{\textcolor{lstOperator}{/}}{1}%
    {\$}{\textcolor{lstOperator}{\$}}{1}%
    {\&}{\textcolor{lstOperator}{\&}}{1}%
    {\^}{\textcolor{lstOperator}{\^}}{1}%
    {\|}{\textcolor{lstOperator}{\|}}{1}%
    {\%\%}{\textcolor{lstOperator}{\%\%}}{2}%
    {\%*\%}{\textcolor{lstOperator}{\%*\%}}{3}%
    {\%/\%}{\textcolor{lstOperator}{\%/\%}}{3}%
    {\%o\%}{\textcolor{lstOperator}{\%o\%}}{3}%
    {\%x\%}{\textcolor{lstOperator}{\%x\%}}{3}%
    {\%in\%}{\textcolor{lstOperator}{\%in\%}}{4},%
  keywords=[1]{% Flow-control reserved words
    break,else,for,if,ifelse,next,repeat,return,switch,while%
  },%
  keywords=[2]{% Other reserved words
    ...,class,in,function,library,require,setClass,setRefClass%
  },%
  keywords=[3]{% Special literals
    FALSE,Inf,NA,NA_character_,NA_complex_,NA_integer_,NA_real_,NaN,NULL,TRUE%
  },%
  keywords=[4]{% Built-in types
    array,character,complex,double,integer,list,logical,LongVectors,matrix,%
    numeric,pairlist,raw,table,vector%
  },%
  keywords=[5]{% Built-in functions
    abs,acos,acosh,agrep,agrepl,all,append,Arg,as.character,as.complex,%
    as.data.frame,as.double,as.environment,as.function,as.hexmode,as.integer,%
    as.list,as.logical,as.matrix,as.numeric,as.pairlist,as.qr,as.raw,as.table,%
    as.vector,asin,asinh,atan,atan2,atanh,attach,attr,basename,by,c,call,%
    casefold,cat,charmatch,charToRaw,chartr,colnames,Conj,cos,cosh,cospi,%
    crossprod,data.frame,data.matrix,date,diff,difftime,dim,dirname,%
    duplicated,eigen,enc2native,enc2utf8,Encoding,environment,exp,expm1,%
    factor,file.access,file.append,file.copy,file.create,file.exists,%
    file.info,file.link,file.mode,file.mtime,file.path,file.remove,%
    file.rename,file.show,file.size,file.symlink,format,getOption,gl,gregexpr,%
    grep,grepl,gsub,iconv,interactive,intToBits,intToUtf8,is.character,%
    is.complex,is.double,is.finite,is.infinite,is.integer,is.list,is.logical,%
    is.matrix,is.na,is.nan,is.numeric,is.pairlist,is.qr,is.raw,is.recursive,%
    is.single,is.table,is.vector,kappa,kronecker,labels,lapply,length,lengths,%
    levels,list.dirs,list.files,load,log,log10,log1p,log2,logb,lower.tri,%
    make.names,make.unique,match,match.arg,max,maxCol,merge,ls,match,mean,%
    message,min,missing,Mod,names,ncol,NCOL,nlevels,norm,normalizePath,NotYet,%
    nrow,NROW,on.exit,options,order,outer,packBits,parent.frame,parse,paste,%
    paste0,path.expand,pchisq,pmatch,pmax,pmax.int,pmin,pmin.int,print,%
    print.data.frame,prmatrix,proc.time,qr,qr.coef,qr.fitted,qr.resid,%
    qr.solve,qr.qty,qr.qy,quit,rank,rapply,rawShift,rawToBits,rawToChar,Re,%
    readLines,regmatches,rep,replace,regexec,regexpr,rev,rle,RNGkind,%
    RNGversion,row,row.names,rownames,sample,sapply,save,scale,scan,seq,%
    set.seed,setSessionTimeLimit,setTimeLimit,sign,sin,sinh,sinpi,solve,%
    source,split,sprintf,sqrt,strsplit,strtoi,strtrim,sub,sum,svd,summary,%
    sys.call,sys.calls,Sys.Date,sys.frame,sys.frames,sys.function,Sys.glob,%
    Sys.getlocale,Sys.junction,Sys.localeconv,sys.nframe,sys.on.exit,%
    sys.parent,sys.parents,Sys.setlocale,Sys.sleep,sys.status,Sys.time,%
    Sys.timezone,system.time,tan,tanh,tanpi,tempfile,tempdir,time,tolower,%
    toupper,transform,try,typeof,unique,unlink,which,write%
  },%
  keywords=[6]{% Operator keywords
    isFALSE,isTRUE,xor%
  },%
  keywords=[7]{% utils
    alarm,available.packages,capture.output,data,flush.console,head,%
    installed.packages,install.packages,read.csv,read.csv2,read.delim,%
    read.delim2,read.table,select.list,write.csv,write.csv2,write.table%
  },%
  keywords=[8]{% graphics, grDevices
    abline,arrows,axis,axTicks,barplot,box,boxplot,clip,close.screen,curve,%
    erase.screen,frame,grid,hist,layout,legend,lines,matplot,mosaicplot,mtext,%
    par,pie,plot,plot.design,plot.new,plot.window,plot.xy,points,rect,screen,%
    segments,spineplot,split.screen,stars,stem,stripchart,symbols,text,title,%
    xinch,xyinch,yinch,%
	axisTicks,colors,colours,dev.capture,dev.cur,dev.flush,dev.list,dev.next,%
    dev.new,dev.off,dev.prev,dev.set,dev.size,graphics.off,gray,gray.colors,%
    grey,grey.colors,palette,pdf,pdf.options,rgb,windows%
  },%
  keywords=[9]{% stats
    acf,add1,AIC,anova,anova.glm,anova.lm,aov,ar,as.ts,asOneSidedFormula,ave,%
    BIC,biplot,biplot.princomp,Box.test,chisq.test,cmdscale,coef,cor,cor.test,%
    cov,cov.wt,cov2cor,cumsum,cutree,cycle,dbeta,dbinom,dcauchy,dchisq,deltat,%
    deviance,df,dgamma,dgeom,dhyper,dist,dlnorm,dlogis,dnorm,drop1,dwilcox,%
    ecdf,effects,expand.model.frame,factanal,fitted,fisher.test,formula,%
    frequency,getInitial,glm,hasTsp,hclust,heatmap,HoltWinters,influence,%
    interaction.plot,IQR,is.ts,kmeans,kruskal.test,ks.test,lag,line,listof,lm,%
    lm.fit,lm.summaries,loadings,loglin,ls.diag,ls.print,lsfit,mad,%
    model.frame,model.matrix,na.action,na.contiguous,na.exclude,na.fail,%
    na.omit,na.pass,optimize,pbeta,pbinom,pcauchy,pchisq,pf,pgamma,pgeom,%
    phyper,plnorm,plogis,pnorm,Poisson,power.anova.test,predict,predict.lm,%
    preplot,pwilcox,qbeta,qbinom,qcauchy,qchisq,qf,qgamma,qgeom,qhyper,qlnorm,%
    qlogis,qnorm,qqline,qqnorm,qqplot,qwilcox,rbeta,rbinom,rcauchy,rchisq,%
    residuals,residuals.glm,rf,rgamma,rgeom,rhyper,rlnorm,rlogis,rnorm,%
    rwilcox,screeplot,sd,t.test,time,ts,tsp,Tukey,TukeyHSD,var,%
    weighted.residuals,weights,Wilcoxon%
  }%
}

%% Appendix numbering
\newif\iffirstappendix
\firstappendixtrue

%% Number to Roman numerals
\newcommand{\Romnum}[1]{\expandafter\@slowromancap\romannumeral #1@}

%% Test if number
\newcommand\ifnumber[1]{%
  \begingroup
    \edef\temp{#1}%
    \expandafter\ifstrempty\expandafter{\temp}
  {\endgroup\@secondoftwo}
  {\expandafter\ifnumber@i\temp\@nnil}%
}
\def\ifnumber@i#1#2\@nnil{%
  \if-#1%
    \ifstrempty{#2}
      {\def\temp{X}}
	  {\def\temp{#2}}%
  \else
    \def\temp{#1#2}%
  \fi
  \afterassignment\ifnumhelper
  \count@0\temp\relax\@nnil
  \endgroup
}
\def\numrelax{\relax}%
\def\ifnumhelper#1\@nnil{%
  \def\temp{#1}%
    \ifx\temp\numrelax
      \aftergroup\@firstoftwo
    \else
      \aftergroup\@secondoftwo
    \fi
}

%% Fonts
\urlstyle{same}
\setverbatimfont{\normalfont}
\renewcommand*{\foottextfont}{\normalfont\normalsize}

%% Document should be double spaced
\DoubleSpacing

%% Footnotes should be ragged right
\renewcommand{\@makefntext}[1]{\makefootmark\raggedright #1}

%% the fiddle lengths (..ta.. for title/approval page, others for prelims)
\newlength{\topafiddle} \setlength{\topafiddle}{50pt} % 2in title with 1in margin
\newlength{\topbfiddle} \setlength{\topbfiddle}{30pt} % 2in title with 98pt margin
\newlength{\topcfiddle} \setlength{\topcfiddle}{-30pt} % 1in title with 1in margin
\newlength{\botafiddle} \setlength{\botafiddle}{50pt}

\setlength{\parindent}{0em}

%%%% pagestyles
%% the main text
\makepagestyle{osu}
\makeevenfoot{osu}{}{\normalfont\normalsize\thepage}{}
\makeoddfoot{osu}{}{\normalfont\normalsize\thepage}{}

%% for continuation pages of the ToC, LoF, LoT
\makepagestyle{toc}
\makeevenfoot{toc}{}{\normalfont\normalsize\thepage}{}
\makeoddfoot{toc}{}{\normalfont\normalsize\thepage}{}
\makeevenhead{toc}{\normalfont\large Chapter}{}{\normalfont\large Page}
\makeoddhead{toc}{\normalfont\large Chapter}{}{\normalfont\large Page}
\makepagestyle{lof}
\makeevenfoot{lof}{}{\lower 6pt\hbox{\normalfont\normalsize\thepage}}{}
\makeoddfoot{lof}{}{\lower 6pt\hbox{\normalfont\normalsize\thepage}}{}
\makeevenhead{lof}{\normalfont\large Figure}{}{\normalfont\large Page}
\makeoddhead{lof}{\normalfont\large Figure}{}{\normalfont\large Page}
\makepagestyle{lot}
\makeevenfoot{lot}{}{\lower 6pt\hbox{\normalfont\normalsize\thepage}}{}
\makeoddfoot{lot}{}{\lower 6pt\hbox{\normalfont\normalsize\thepage}}{}
\makeevenhead{lot}{\normalfont\large Table}{}{\normalfont\large Page}
\makeoddhead{lot}{\normalfont\large Table}{}{\normalfont\large Page}

%%%% Do the ToC
\cftsetindents{chapter}{0em}{2em}
\cftsetindents{section}{2em}{2.4em}
\cftsetindents{subsection}{4.4em}{3.2em}

\addto\captionsenglish{\renewcommand{\contentsname}{{\large TABLE OF CONTENTS}}}
\newcommand{\tocgeometry}{\newgeometry{left=1.5in,right=1in,top=98pt,bottom=1.4in,headheight=23pt,headsep=13pt,footskip=.5in}}
\addtodef{\tableofcontents}%
  {\clearpage\pagestyle{toc}\begin{KeepFromToc}\tocgeometry\SingleSpacing}%
  {\restoregeometry\end{KeepFromToc}\restoregeometry\clearpage\DoubleSpacing}
\renewcommand*{\tocheadstart}{\vspace*{\topbfiddle}}
\renewcommand*{\aftertoctitle}{\vspace*{2\baselineskip}\thispagestyle{plain}%
  \par\nobreak {\normalfont\large Chapter \hfill Page}\par\nobreak}

\renewcommand*{\cftchapterfont}{\normalfont\large}
\renewcommand*{\cftchapterpagefont}{\normalfont\large}
\renewcommand\cftsectionfont{\normalfont\large}
\renewcommand\cftsectionpagefont{\normalfont\large}
\renewcommand\cftsubsectionfont{\normalfont\large}
\renewcommand\cftsubsectionpagefont{\normalfont\large}
\renewcommand*{\cftchapterleader}{%
  \cftchapterfont\cftdotfill{\cftchapterdotsep}}
\renewcommand*{\cftchapterdotsep}{\cftdotsep}
\setlength{\cftbeforechapterskip}{1\onelineskip \@plus\p@}
\renewcommand*{\insertchapterspace}{}

\patchcmd{\l@chapapp}%
  {\cftchapterfont {#1}}%
  {\normalfont \texorpdfstring{%
    \uppercase{#1}}{{#1}} }%
  {\typeout{Success: Patch %
    'l@chapapp' to uppercase %
    part-level headings in the %
    table of contents.}}%
  {\typeout{Fail: Patch %
    'l@chapapp' to uppercase %
    part-level headings in the %
    table of contents.}}%

\renewcommand{\chapternumberline}[1]{%
  \chapternumberlinehook{#1}%
  \chapternumberlinebox\@tempdima{\@chapapp@head\@cftbsnum \ifnumber{#1}{\Romnum{#1}}{#1}\@cftasnum}%
  \@cftasnumb}

\setsechook{%
  \ifnum\value{section}=0%
    \addtocontents{toc}{\protect\addvspace{\onelineskip}}%
  \fi}

\maxtocdepth{subsection}

%%% The LoT
\addtodef{\listoftables}%
  {\clearpage\pagestyle{lot}\begin{KeepFromToc}\tocgeometry\SingleSpacing}%
  {\restoregeometry\end{KeepFromToc}\restoregeometry\clearpage\DoubleSpacing}
\renewcommand{\listtablename}{\large LIST OF TABLES}
\renewcommand*{\lotheadstart}{\vspace*{\topbfiddle}}
\renewcommand*{\afterlottitle}{\vspace*{2\baselineskip}\thispagestyle{plain}%
  \par\nobreak {\normalfont\large Table \hfill Page}\par\nobreak}
\renewcommand*{\cftfigurefont}{\normalfont\large}
\renewcommand*{\cftfigurepagefont}{\normalfont\large}
\setlength\cftbeforetableskip{\cftbeforechapterskip}

%%%% The LoF
\addtodef{\listoffigures}%
  {\clearpage\pagestyle{lof}\begin{KeepFromToc}\tocgeometry\SingleSpacing}%
  {\restoregeometry\end{KeepFromToc}\restoregeometry\clearpage\DoubleSpacing}
\renewcommand{\listfigurename}{\large LIST OF FIGURES}
\renewcommand*{\lofheadstart}{\vspace*{\topbfiddle}}
\renewcommand*{\afterloftitle}{\vspace*{2\baselineskip}\thispagestyle{plain}%
  \par\nobreak {\normalfont\large Figure \hfill Page}\par\nobreak}
\renewcommand*{\cfttablefont}{\normalfont\large}
\renewcommand*{\cfttablepagefont}{\normalfont\large}
\setlength\cftbeforefigureskip{\cftbeforechapterskip}

%% chapter style
\makechapterstyle{osu}{%
  \setlength{\beforechapskip}{\topafiddle}
  \setlength{\midchapskip}{40pt}
  \setlength{\afterchapskip}{36pt}
  \renewcommand*{\chapnamefont}{\normalfont}
  \renewcommand*{\chapnumfont}{\chapnamefont}
  \renewcommand*{\printchaptername}{\chapnamefont\large\MakeUppercase{\@chapapp}}
  \renewcommand{\printchapternum}{\centering\chapnumfont\ifanappendix\thechapter\else\Roman{chapter}\fi}
  \renewcommand*{\chaptitlefont}{\normalfont\centering}
  \renewcommand{\printchaptertitle}[1]{\chaptitlefont\normalsize\MakeUppercase{##1}\par\nobreak}
  \abnormalparskip{1\onelineskip}\raggedright}
\aliaspagestyle{chapter}{plain}

\renewcommand{\mempreaddchaptertotochook}{%
	\ifanappendix\iffirstappendix
	\firstappendixfalse
	\addcontentsline{toc}{chapter}{APPENDICES}
	\fi\fi}

%%% (sub)section styles
\setsecheadstyle{\normalfont\SingleSpacing}
\setbeforesecskip{48pt plus -.8ex minus -.2ex}
\setaftersecskip{15pt plus .2ex}
\setsubsecheadstyle{\normalfont\SingleSpacing}
\setbeforesubsecskip{15pt plus -.6ex minus -.2ex}
\setaftersubsecskip{15pt plus .2ex}
\setsecnumdepth{subsection}

%% Captions
\captiontitlefont{\normalfont\small\OnehalfSpacing}
\precaption{\raggedright}
\captionstyle{\raggedright}
\captiontitlefinal{}
\setlength{\belowcaptionskip}{.5\onelineskip}

%% for REFERENCES section
\setlength\bibitemsep{\onelineskip}
%\renewcommand{\bibname}{References}
%\addto{\captionsenglish}{\renewcommand{\refname}{References}}
\AtBeginBibliography{\normalfont\normalsize\raggedright\SingleSpacing}
%\renewcommand*{\bibname}{REFERENCES}
%\setlength\bibhang{2em}
%setlength{\bibitemsep}{\onelineskip}
%\renewcommand*{\biblistextra}{%
%\setlength{\itemsep}{\bibitemsep}}
%\AtBeginEnvironment{thebibliography}{\normalfont\normalsize\SingleSpacing}

%%%% for the title page and approval page.
\newcommand{\settitle}[1]{\def\osutitle{#1}}
\newcommand{\setabstracttitle}[1]{\def\osuabstracttitle{#1}}
\newcommand{\setauthor}[1]{\def\osuauthor{#1}}
\newcommand{\setmajor}[1]{\def\osumajor{#1}}
\newcommand{\setdegree}[1]{\def\osudegree{#1}}
\newcommand{\setdegreeOne}[1]{\def\osudegreeOne{#1}}
\newcommand{\setdegreeTwo}[1]{\def\osudegreeTwo{#1}}
\newcommand{\setgraddate}[1]{\def\osugraddate{#1}}
\newcommand{\setadvisor}[1]{\def\osuadvisor{#1}}
\newcommand{\setmemberOne}[1]{\def\osumemberOne{#1}}
\newcommand{\setmemberTwo}[1]{\def\osumemberTwo{#1}}
\newcommand{\setmemberThree}[1]{\def\osumemberThree{#1}}
\newcommand{\setmemberFour}[1]{\def\osumemberFour{#1}}
\newcommand{\setmemberOutside}[1]{\def\osumemberOutside{#1}}

%%% typesets the TITLE page
\newcommand{\thetitlepage}{{%
  \clearpage
  \thispagestyle{empty}
  \abnormalparskip{0em}
  \centering\large
  \vspace*{\topafiddle}
  \MakeUppercase{\osutitle}\par
  \vfill By \\
  \MakeUppercase{\osuauthor}\par
  \ifdefined\osudegreeOne
    {\SingleSpacing \osudegreeOne\par}
  \fi
  \ifdefined\osudegreeTwo
    {\SingleSpacing \osudegreeTwo\par}
  \fi
  \vfill
  {\SingleSpacing Submitted to the Faculty of the \\
    Graduate College of the \\
    Oklahoma State University \\
    in partial fulfillment of \\
    the requirements for \\
    the Degree of \\
    \MakeUppercase{\osudegree} \\
    \osugraddate\par}
  \vspace*{\botafiddle}
  \clearpage}}

%%% typesets the APPROVAL page
\newcommand{\approvalpage}{{%
  \clearpage
  \thispagestyle{plain}
  \abnormalparskip{0em}
  \centering{\large
  \vspace*{\topafiddle}
  \MakeUppercase{\osutitle}\par}
  \vspace*{56pt}
  {\raggedright\hspace*{1in}Dissertation Approved:\par}
  \vspace*{27pt}
  \osuadvisor \\[-.3\baselineskip]
  {\raggedright\hspace*{1in}\rule{4in}{.4pt}\par}
  \vspace*{-11pt}
  Dissertation Adviser \\[.1\baselineskip]
  \osumemberOne \\[-.3\baselineskip]
  {\raggedright\hspace*{1in}\rule{4in}{.4pt}\par}
  \vspace*{.6\baselineskip}
  \osumemberTwo \\[-.3\baselineskip]
  {\raggedright\hspace*{1in}\rule{4in}{.4pt}\par}
  \ifdefined\osumemberThree
    \vspace*{.6\baselineskip}
    \osumemberThree \\[-.3\baselineskip]
    {\raggedright\hspace*{1in}\rule{4in}{.4pt}\par}
  \fi
  \ifdefined\osumemberFour
    \vspace*{.6\baselineskip}
    \osumemberFour \\[-.3\baselineskip]
    {\raggedright\hspace*{1in}\rule{4in}{.4pt}\par}
  \fi
  \ifdefined\osumemberOutside
    \vspace*{.6\baselineskip}
    \osumemberOutside \\[-.3\baselineskip]
    {\raggedright\hspace*{1in}\rule{4in}{.4pt}\par}
  \fi
  \clearpage}}

%%% typesets the ACKNOWLEDGEMENTS page
\newcommand{\blindfootnote}[1]{%
  \begingroup
  \renewcommand\thefootnote{}\footnote{#1}%
  \addtocounter{footnote}{-1}%
  \endgroup}
\newenvironment{acknowledgementspage}{%
  \clearpage
  \thispagestyle{plain}
  \abnormalparskip{0em}
  \vspace*{\topafiddle}\large
  {\centering ACKNOWLEDGEMENTS\par\vspace*{\onelineskip}}
  \SingleSpacing}{%
  \blindfootnote{Acknowledgements reflect the views of the author and are not
  endorsed by committee members or Oklahoma State University.}\par\clearpage}

%%% typesets the ABSTRACT page
\newenvironment{abstractpage}{%
  \clearpage
  \thispagestyle{plain}
  \abnormalparskip{0em}
  \raggedright\large
  \vspace*{\topafiddle}
  Name: \MakeUppercase{\osuauthor}\par
  Date of Degree: \MakeUppercase{\osugraddate}\par
  \newlength{\titleofstudy}
  \settowidth{\titleofstudy}{Title of Study:\hspace*{1pt}\ }
  {\SingleSpacing\hangpara{\titleofstudy}{1}Title of Study: \MakeUppercase{\osuabstracttitle}\par}
  Major Field: \MakeUppercase{\osumajor}\par
  \SingleSpacing Abstract:}{%
  \par\clearpage}

%%% typesets the VITA page
\newenvironment{vitapage}{%
  \clearpage
  \thispagestyle{empty}
  \SingleSpacing
  \raggedright\large
  \vspace*{\topcfiddle}
  {\centering \osuauthor \\[1\baselineskip]
  Candidate for the Degree of \\[1\baselineskip]
  \osudegree \\[1\baselineskip]}
  \newlength{\thesiswidth}
  \settowidth{\thesiswidth}{Thesis:\hspace*{1pt}\ }
  \hangpara{\thesiswidth}{1}Thesis: \MakeUppercase{\osuabstracttitle}\par
  \vspace*{1\onelineskip}
  Major Field: \osumajor\par
  Biographical:\par}{%
  \par\clearpage}

\newenvironment{education}{%
  \hfill\begin{minipage}{0.9\textwidth}\raggedright Education:\par
  Completed the requirements for the \osudegree\ in \osumajor\ at Oklahoma State University, Stillwater, Oklahoma in \osugraddate.\par}{%
  \end{minipage}\par}

\newenvironment{experience}{%
  \hfill\begin{minipage}{0.9\textwidth}\raggedright Experience:\par}{%
  \end{minipage}\par}

\newenvironment{memberships}{%
  \hfill\begin{minipage}{0.9\textwidth}\raggedright Professional Memberships:\par}{%
  \end{minipage}}

%% Listings minus sign replacement
\makeatletter
\lst@CCPutMacro\lst@ProcessOther {"2D}{\lst@ttfamily{-{}}{-{}}}
\@empty\z@\@empty
\makeatother

%% Align to top on floats only pages
\makeatletter
\setlength{\@fptop}{0pt}
\makeatother

%% Paths
\newcommand{\setfigurepath}[1]{\def\osufigurepath{#1}}
\newcommand{\setlistingpath}[1]{\def\osulistingpath{#1}}
\setfigurepath{.}
\setlistingpath{.}

%% Tables
\newenvironment{singletable}[3]{%
  \begin{table}[t]
    \centering
    \caption[#2]{#3}
    \label{#1}}{%
  \end{table}}

%% Listings
\newcommand{\pylisting}[3]{
  \begin{itemize}[leftmargin=1.2em,rightmargin=.2em,itemsep=0pt,labelsep=0pt]
    \SingleSpacing
    \item[]\lstinputlisting[language=PythonPlus,label=#2,caption=#3]{\osulistingpath/#1}
  \end{itemize}}

\newcommand{\rlisting}[3]{
  \begin{itemize}[leftmargin=1.2em,rightmargin=.2em,itemsep=0pt,labelsep=0pt]
    \SingleSpacing
    \item[]\lstinputlisting[language=RPlus,label=#2,caption=#3]{\osulistingpath/#1}
  \end{itemize}}

%% Algorithms
\newenvironment{singlealgorithm}[2]{%
  \begin{algorithm}[ht]
    \centering
    \caption{#2}
    \label{#1}
    \begin{algorithmic}[1]}{%
    \end{algorithmic}
  \end{algorithm}%
  \vspace{-13pt}}

%% Figures
\newcommand{\singlefigure}[5][\textwidth]{%
  \begin{figure}[t]
    \centering
    \begin{minipage}[t]{#1}
      \vspace{0pt}
      \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/#3}}
    \end{minipage}
    \caption[#4]{#5}
    \label{#2}
  \end{figure}}

\newcommand{\hdoublefigure}[7][\textwidth]{%
  \begin{figure}[t]
  	\centering
    \begin{minipage}[t]{#1}
      \begin{minipage}[t]{.5\textwidth-4pt}
        \begin{minipage}[t]{\widthof{(a)\,}}
          \vspace{2pt}(a)
        \end{minipage}%
        \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
          \vspace{0pt}
          \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/#3}}
        \end{minipage}%
        \vspace{0pt}
      \end{minipage}%
      \hspace*{4pt}
      \begin{minipage}[t]{.5\textwidth-4pt}
        \begin{minipage}[t]{\widthof{(a)\,}}
          \vspace{2pt}(b)
        \end{minipage}%
        \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
          \vspace{0pt}
          \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/#4}}
        \end{minipage}%
        \vspace{0pt}
      \end{minipage}%
      \hfill
    \end{minipage}
    \caption[#5]{(a)~#6\,\, (b)~#7}
    \label{#2}
  \end{figure}}

\newcommand{\vdoublefigure}[7][\textwidth]{%
  \begin{figure}[t]
  	\centering
    \begin{minipage}[t]{#1}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{\widthof{(a)\,}}
          \vspace{2pt}(a)
        \end{minipage}%
        \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
          \vspace{0pt}
          \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/#3}}
        \end{minipage}%
        \vspace{0pt}
      \end{minipage}%
      \hfill\vspace{-4pt}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{\widthof{(a)\,}}
          \vspace{2pt}(b)
        \end{minipage}%
        \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
          \vspace{0pt}
          \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/#4}}
        \end{minipage}%
        \vspace{0pt}
      \end{minipage}%
      \hfill
    \end{minipage}
    \caption[#5]{(a)~#6\,\, (b)~#7}
    \label{#2}
  \end{figure}}

\newcommand{\triplefigure}[9][\textwidth]{%
  \begin{figure}[t]
  	\centering
    \begin{minipage}[t]{#1}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{\widthof{(a)\,}}
          \vspace{2pt}(a)
        \end{minipage}%
        \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
          \vspace{0pt}
          \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/#3}}
        \end{minipage}%
        \vspace{0pt}
      \end{minipage}%
      \hfill\vspace{-4pt}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{\widthof{(a)\,}}
          \vspace{2pt}(b)
        \end{minipage}%
        \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
          \vspace{0pt}
          \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/#4}}
        \end{minipage}%
        \vspace{0pt}
      \end{minipage}%
      \hfill\vspace{-4pt}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{\widthof{(a)\,}}
          \vspace{2pt}(c)
        \end{minipage}%
        \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
          \vspace{0pt}
          \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/#5}}
        \end{minipage}%
        \vspace{0pt}
      \end{minipage}%
      \hfill
    \end{minipage}
    \caption[#6]{(a)~#7\,\, (b)~#8\,\, (c)~#9}
    \label{#2}
  \end{figure}}

\newcommand{\quadfigure}[7][\textwidth]{%
  \def\tempa{#1}%
  \def\tempb{#2}%
  \def\tempc{#3}%
  \def\tempd{#4}%
  \def\tempe{#5}%
  \def\tempf{#6}%
  \def\tempg{#7}%
  \quadfigurecontinued}
\newcommand{\quadfigurecontinued}[4]{%
  \begin{figure}[t]
  	\centering
    \begin{minipage}[t]{\tempa}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(a)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\tempc}}
          \end{minipage}
          \vspace{0pt}
        \end{minipage}%
        \hspace*{4pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(b)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\tempd}}
          \end{minipage}
          \vspace{0pt}
        \end{minipage}%
	    \hfill
      \end{minipage}%
      \hfill\vspace{-4pt}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(c)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\tempe}}
          \end{minipage}%
          \vspace{0pt}
        \end{minipage}%
        \hspace*{4pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(d)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\tempf}}
          \end{minipage}%
          \vspace{0pt}
        \end{minipage}%
	    \hfill
      \end{minipage}%
	  \hfill
    \end{minipage}
    \caption[\tempg]{(a)~#1\,\, (b)~#2\,\, (c)~#3\,\, (d)~#4}
    \label{\tempb}
  \end{figure}}

\newcommand{\sixfigure}[9][\textwidth]{%
  \def\tempa{#1}%
  \def\tempb{#2}%
  \def\tempc{#3}%
  \def\tempd{#4}%
  \def\tempe{#5}%
  \def\tempf{#6}%
  \def\tempg{#7}%
  \def\temph{#8}%
  \def\tempi{#9}%
  \sixfigurecontinued}
\newcommand{\sixfigurecontinued}[6]{%
  \begin{figure}[t]
  	\centering
    \begin{minipage}[t]{\tempa}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(a)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\tempc}}
          \end{minipage}
          \vspace{0pt}
        \end{minipage}%
        \hspace*{4pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(b)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\tempd}}
          \end{minipage}
          \vspace{0pt}
        \end{minipage}%
	    \hfill
      \end{minipage}%
      \hfill\vspace{-4pt}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(c)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\tempe}}
          \end{minipage}%
          \vspace{0pt}
        \end{minipage}%
        \hspace*{4pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(d)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\tempf}}
          \end{minipage}%
          \vspace{0pt}
        \end{minipage}%
	    \hfill
      \end{minipage}%
      \hfill\vspace{-4pt}
      \begin{minipage}[t]{\textwidth-2pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(e)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\tempg}}
          \end{minipage}%
          \vspace{0pt}
        \end{minipage}%
        \hspace*{4pt}
        \begin{minipage}[t]{.5\textwidth-4pt}
          \begin{minipage}[t]{\widthof{(a)\,}}
            \vspace{2pt}(f)
          \end{minipage}%
          \begin{minipage}[t]{\textwidth-\widthof{(a)\,}}
            \vspace{0pt}
            \parbox{\textwidth}{\includegraphics[width=\textwidth]{\osufigurepath/\temph}}
          \end{minipage}%
          \vspace{0pt}
        \end{minipage}%
	    \hfill
      \end{minipage}%
	  \hfill
    \end{minipage}
    \caption[\tempi]{(a)~#1\,\, (b)~#2\,\, (c)~#3\,\, (d)~#4\,\, (e)~#5\,\, (f)~#6}
    \label{\tempb}
  \end{figure}}

%% Specify chapter and page styles
\chapterstyle{osu}
\pagestyle{plain}

\endinput
